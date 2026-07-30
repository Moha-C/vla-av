#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import sys
import time
import traceback

from PIL import Image


PROMPT = """You are an external VLM-CoT observer for a CARLA autonomous-driving demo.
Look only at the camera image and reason conservatively about the driving scene.
You must NOT output steering, throttle, brake, or any low-level control.
Return ONLY valid JSON with these keys:
{
  "scene_type": "normal|blocked_lane|traffic_light|pedestrian|vehicle_cut_in|traffic_jam|unknown",
  "risk_level": "low|medium|high|critical|unknown",
  "main_hazard": "short phrase",
  "safe_hint": "short semantic hint for the driving stack",
  "reason": "one concise natural-language reasoning sentence",
  "confidence": 0.0
}
Critical rules:
- Be conservative. If you are unsure, use risk_level "medium" and confidence <= 0.55.
- Do NOT say "normal" or "low" if there is any vehicle close ahead, beside the ego, partially blocking the lane, merging, stopped, crashed, or overlapping another vehicle.
- A close vehicle directly ahead is at least "vehicle_ahead" with risk "medium", even if it is moving.
- A blocked ego lane, stopped vehicle, police/accident vehicle, debris, or vehicle across lane markings is "blocked_lane" with risk "high".
- If a lane change or overtake may be needed, mention that it is only safe after checking adjacent/oncoming traffic.
Focus on the ego lane, the next 30 meters, adjacent lanes, stopped vehicles, side collisions, pedestrians, traffic lights, and oncoming traffic."""


def atomic_write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, payload):
    if not path:
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def base_status(args, event="cot_update"):
    return {
        "event": event,
        "timestamp": time.time(),
        "mode": args.mode,
        "model": args.model,
        "scene_type": "unknown",
        "risk_level": "unknown",
        "main_hazard": "none",
        "safe_hint": "observe_only",
        "reason": "Waiting for camera analysis.",
        "confidence": 0.0,
        "control_authority": "none",
    }


def extract_json(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in output: {text[:240]}")
    return json.loads(match.group(0))


def normalize_payload(payload, args, raw_text="", inference_seconds=0.0):
    status = base_status(args)
    status.update({
        "scene_type": str(payload.get("scene_type", "unknown"))[:64],
        "risk_level": str(payload.get("risk_level", "unknown"))[:32],
        "main_hazard": str(payload.get("main_hazard", "unknown"))[:120],
        "safe_hint": str(payload.get("safe_hint", "observe_only"))[:160],
        "reason": str(payload.get("reason", ""))[:360],
        "raw_text": str(raw_text)[:1000],
        "inference_seconds": round(float(inference_seconds), 3),
    })
    try:
        status["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except Exception:
        status["confidence"] = 0.0
    text = " ".join(
        str(status.get(key, ""))
        for key in ("scene_type", "risk_level", "main_hazard", "safe_hint", "reason", "raw_text")
    ).lower()
    vehicle_ahead_terms = (
        "vehicle ahead",
        "car ahead",
        "car in front",
        "vehicle in front",
        "safe distance",
        "following distance",
    )
    blocked_terms = (
        "blocked",
        "stopped vehicle",
        "accident",
        "crash",
        "collision",
        "obstruction",
        "police",
        "debris",
    )
    if any(term in text for term in blocked_terms):
        if status["risk_level"] in ("low", "unknown"):
            status["risk_level"] = "high"
        if status["scene_type"] in ("normal", "unknown"):
            status["scene_type"] = "blocked_lane"
        status["confidence"] = min(status["confidence"], 0.75)
        status["conservative_adjustment"] = "blocked_or_accident_terms_detected"
    elif any(term in text for term in vehicle_ahead_terms):
        if status["risk_level"] in ("low", "unknown"):
            status["risk_level"] = "medium"
        if status["scene_type"] in ("normal", "unknown"):
            status["scene_type"] = "vehicle_ahead"
        status["confidence"] = min(status["confidence"], 0.65)
        status["conservative_adjustment"] = "vehicle_ahead_terms_detected"
    return status


def mock_reason(frame_path, args):
    with Image.open(frame_path) as image:
        width, height = image.size
    return normalize_payload(
        {
            "scene_type": "unknown",
            "risk_level": "unknown",
            "main_hazard": "mock observer",
            "safe_hint": "vlm_cot_mock_only",
            "reason": (
                f"Mock CoT is receiving CARLA camera frames ({width}x{height}) "
                "but no VLM inference is active."
            ),
            "confidence": 0.0,
        },
        args,
        raw_text="mock",
    )


class Qwen2VLBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.args = args
        self.torch = torch
        self.device = args.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and args.device == "auto":
            raise RuntimeError("CUDA is not available; refusing to load a 7B VLM on CPU in auto mode")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": args.local_files_only,
            "trust_remote_code": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def infer(self, frame_path):
        image = Image.open(frame_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        started = time.time()
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens)
        generated_ids = [
            output_ids_item[len(input_ids):]
            for input_ids, output_ids_item in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        payload = extract_json(output_text)
        return normalize_payload(
            payload,
            self.args,
            raw_text=output_text,
            inference_seconds=time.time() - started,
        )


def load_backend(args):
    if args.mode in ("off", ""):
        return None
    if args.mode == "mock":
        return "mock"
    if args.mode in ("qwen2_vl", "auto"):
        return Qwen2VLBackend(args)
    raise ValueError(f"Unsupported CoT mode: {args.mode}")


def write_error(args, message, detail=""):
    status = base_status(args, event="cot_error")
    status.update({
        "scene_type": "unknown",
        "risk_level": "unknown",
        "main_hazard": "vlm_cot_error",
        "safe_hint": "cot_unavailable_observe_only",
        "reason": message[:360],
        "confidence": 0.0,
        "error": detail[:2000],
    })
    atomic_write_json(args.status_path, status)
    append_jsonl(args.log_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="off", choices=["off", "mock", "auto", "qwen2_vl"])
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--frame-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fallback-mock", action="store_true")
    args = parser.parse_args()

    if args.mode == "off":
        return 0

    atomic_write_json(args.status_path, base_status(args, event="cot_starting"))
    try:
        backend = load_backend(args)
    except Exception as exc:
        detail = traceback.format_exc()
        write_error(args, f"Could not load external VLM-CoT backend: {exc}", detail)
        if not args.fallback_mock:
            return 2
        backend = "mock"

    frame_path = pathlib.Path(args.frame_path)
    last_mtime = None
    while True:
        try:
            if not frame_path.exists():
                time.sleep(0.25)
                continue
            mtime = frame_path.stat().st_mtime
            if last_mtime is not None and mtime <= last_mtime:
                time.sleep(0.1)
                continue
            last_mtime = mtime

            if backend == "mock":
                status = mock_reason(frame_path, args)
            else:
                status = backend.infer(frame_path)
            atomic_write_json(args.status_path, status)
            append_jsonl(args.log_path, status)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            write_error(args, f"VLM-CoT inference failed: {exc}", traceback.format_exc())
        time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    sys.exit(main())
