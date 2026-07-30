import asyncio
import sys
from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("Test Agent sender")

@fast.agent(name ="Agent Sender",servers=["streamable_http_server"], instruction="You are a helpful assistant")

async def main(prompt):
    async with fast.run() as agent:
        return await agent.send(prompt)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        print(asyncio.run(main(prompt)))

