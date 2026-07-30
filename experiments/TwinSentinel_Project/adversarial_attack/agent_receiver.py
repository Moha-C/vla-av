import asyncio
import sys
from mcp_agent.core.fastagent import FastAgent

#Variables
global first_prompt, last_prompt

fast = FastAgent("Test Agent sender")

@fast.agent(name ="Agent Sender", instruction="You are a helpful assistant")

async def main(prompt):
    global first_prompt, last_prompt
    async with fast.run() as agent:
        first_prompt= await agent.send(prompt)
        last_prompt = await agent.send("Call test endpoint.")

if __name__ == "__main__":
    global last_prompt
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        asyncio.run(main(prompt))
        print("the last prompt is :",last_prompt)

