Just a general page for understanding some of the code in this 
directory. 

---
`rollout.py`

- This file implements the basic functionality to save videos of
    the simulation.
    - Here are the functions
      - `def rollout()` saves images 
      - `def render_rollout()` saves videos
    - These functions use the environment method `render()` which
      is inherited from the `class MapEnv` in `map_env.py` which returns 
      an image of the environment's current state. 

