Compiler Architecture
=====================

MechanicsDSL uses a multi-pass compiler architecture designed for safety and extensibility.

Pipeline Overview
-----------------

1.  **Frontend (Lexer/Parser)**: 
    * Input: Raw DSL string.
    * Output: Abstract Syntax Tree (AST).
    * *Security*: No ``eval()`` is used. The parser is a recursive descent implementation.

2.  **Middle-End (Semantic Analysis)**:
    * Input: AST.
    * Output: Symbol Table, Particle Lists.
    * *Logic*: Determines if the system is rigid body (ODEs) or fluid (Particles).

3.  **Backend (Symbolic Engine)**:
    * Input: Symbol Table.
    * Output: Simplified SymPy Equations.
    * *Optimization*: Common subexpression elimination (CSE) happens here.

4.  **Codegen (Transpiler)**:
    * Input: Equations + Particles.
    * Output: C++17 Source Code.
    * *Strategy*: Template injection based on system type.
