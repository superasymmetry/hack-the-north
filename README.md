# ~ Let me do it for you ~

## Inspiration
Google Deepmind's SIMA 2: https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/
and Thinking Machines Lab's interaction models: https://thinkingmachines.ai/blog/interaction-models/ 

## What it does
~ Does things for you ~ (let me do it for you)
Plays your minecraft for you in real time, while roasting you in the process.

## How we built it
I have a lot of GPUs through my uni. Through inferencing open-source models on vLLM with continuous batching in different threads, the agent could achieve an illusion of real-time interaction.

## Challenges we ran into
- fast communication between the local client and remote GPUs
- had to set up cloudflare tunnel
- model for short horizon task was inaccurate

## Accomplishments that we're proud of
- this project is super funny

## What we learned
- never let a doggo do something for you

## What's next for Let me do it for you
- do it yourself
