### Removing `Ray`

At the moment, the base environment class inherits from Ray's `MultiAgentEnv` class, but I think it would be a lot 
cleaner if I was to just build the environment directly for PettingZoo's `ParallelEnv` class. 

I would need to test the performance of the above of course.

Also note that installing `ray`, or at least `ray[rllib]` brings with it a lot more dependencies. It adds about 32 
libraries (the `requirements.txt` file went from 19 to 51 lines after pip installing it)


### Priorities

- [] Get the ensv passing all tests.



## Log 

### 15/07/22

**Passing the pettingZoo tests**

- Changed `aec_to_parallel_wrapper` to `aec_to_parallel_wrapper` (I mixed them up the first time!)
  - Note that I changed from `from_parallel_wrapper` to these yesterday as that function is now deprecated. 

- 