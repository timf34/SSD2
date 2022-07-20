### Setup

At the moment, things will only work using WSL2 as Ubuntu seems to allow me to installing packages that conflict and 
don't work in Windows. I was lead to do this as thankfully one of the developers published there setup on a GitHub 
issue (here)[https://github.com/Farama-Foundation/PettingZoo/issues/710#issuecomment-1141321736]. 

I still need to install `torch` properly into the venv (from source so it works with cuda), but for the moment, it is 
installed properly in my base WSL2 so I will just work ahead with that for now 


### Removing `Ray`

At the moment, the base environment class inherits from Ray's `MultiAgentEnv` class, but I think it would be a lot 
cleaner if I was to just build the environment directly for PettingZoo's `ParallelEnv` class. 

I would need to test the performance of the above of course.

Also note that installing `ray`, or at least `ray[rllib]` brings with it a lot more dependencies. It adds about 32 
libraries (the `requirements.txt` file went from 19 to 51 lines after pip installing it)


### Priorities

- [] Get the ensv passing all tests.



## Log 

### 20/7/22

**Integrated proper wandb logging; installed torch with cuda on WSL2.**

- Cleaned up `sb3_train.py`
  - Made a new config dataclass 
- Installed torch with cuda
  - This was a hassle on WSL2, and I don't think the right version is installed but it runs
  - I will need to check if this does cause issues though 
- Added wandb logging 

### 19/07/22 

**Integrating wandb;**

- Merged `updating_to_new_pettingzoo` into main and archived/ deleted it.
  - Link here: https://stackoverflow.com/questions/1307114/how-can-i-archive-git-branches

### 18/07/22

**Environments now inherit from PettingZoo and work**

- Dependency conflicts will work on WSL2
- I updated the environments to now be compatible with PettingZoo's ParallelEnv insteaf of Rays MultiAgentEnv.
  - Changed `self.num_agents` to `self._num_agents` to avoid base method conflict 
  - Added `self.start` attribute to `class DiscreteWithDType`
  - Changed the callback from DefaultCallbacks (ray) to BaseCallbacks (pettingzoo) 
  
### 15/07/22

**Passing the pettingZoo tests**

- Changed `aec_to_parallel_wrapper` to `aec_to_parallel_wrapper` (I mixed them up the first time!)
  - Note that I changed from `from_parallel_wrapper` to these yesterday as that function is now deprecated.


**Integrating SB3**

- Note to not specify a specific version for `gym` or `supersuit`in the `requirements.txt` file to avoid conflicts.  