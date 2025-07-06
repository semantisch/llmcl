# BilAI Summer School 2025 Project

The goal of this group task is to automatically solve logic puzzles, such as the well-known [Zebra Puzzle](https://en.wikipedia.org/wiki/Zebra_Puzzle) or its variants on websites like [Logic Grid Puzzles](https://daydreampuzzles.com/logic-grid-puzzles/) or [Puzzle Baron's Logic Puzzles](https://logic.puzzlebaron.com/), by means of symbolic and sub-symbolic AI methods. The [general task description](./group-project-3.pdf) explains the addressed problem in more detail.

We hope to solve the problem by combining Large Language Models (LLMs) with Answer Set Programming (ASP). Symbolic ASP methods are effective to encode general domain knowledge in a general, instance-independent way. See, for instance, the [Answer Set Solving in Practice](https://teaching.potassco.org/) course on how to solve application problems by using the ASP system [clingo](https://potassco.org/clingo/). Some logic puzzles instances that have been written by hand are available in the [instances folder](./instances/README), and an incomplete general problem encoding is supplied in the [encodings folder](./encodings/README).

Rather than specifying the ASP facts representing logic puzzle instances by hand,
sub-symbolic LLMs can be utilized to automate the process. A respective framework has recently been introduced in the paper [Integrating Answer Set Programming and Large Language Models for Enhanced Structured Representation of Complex Knowledge in Natural Language](https://tinyurl.com/ijcai25-llmasp), presenting the [LLMASP framework](https://github.com/lewashby/llmasp).

The overall group task thus consists of two parts:

1. Development of a general ASP encoding characterizing solutions to logic puzzle instances specified by ASP facts.
2. Development of an approach using LLMs to turn natural language specifications of logic puzzle instances into ASP facts.

By combining the symbolic and sub-symbolic AI methods, the process of solving of logic puzzles shall be automated to a high degree, saving time for specifying the inputs to symbolic methods and increasing the accuracy over sub-symbolic methods.
