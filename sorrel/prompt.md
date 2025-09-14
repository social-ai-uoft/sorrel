I want to convert this game into the sorrel framework (should be placed under sorrel/examples/)
1. Read the uploaded game description file.
2. Generate a detailed description of the core mechanics, agents, environment, and interactions in this game.
3. Using the sorrel repo (treasurehunt and cleanup and staghunt examples) as reference, create a design specification mapping this game into sorrel’s structure.
   - Directory structure: agents/, entities/, env/, world/, assets/, main.py
   - Internal structure: env should have setup_agents() and populate_environment(); main.py should follow the format in treasurehunt/cleanup/staghunt (no training loop, just experiment.run_experiment()).
   - Interaction mechanisms: make sure beams/interactions match staghunt conventions.
4. Write the sorrel-style implementation of the game based on the design spec.
5. Compare your implementation against treasurehunt.zip and staghunt.zip to ensure structural similarity (both file-level and inside-file organization).
6. Revise until the style and structure are consistent with sorrel.
7. Check coverage: verify that all important features of the original game are preserved.
