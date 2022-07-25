# This is a rough Python file to just help me with estimating how long a full training round would take.

# Not clean but does the job.

def seconds_to_minutes(seconds:int) -> float:
    return seconds / 60


def seconds_to_hours(seconds: int) -> float:
    return seconds / 3600


def estimate_time_for_full_training_runs(environment_steps: int, seconds: int) -> None:
    """
    Given the number of environment_steps completed, and the number of seconds it took to complete those episodes,
    estimate how long it would take to run a full training round of 1e8 to 5e8

    This assumes the speed would be generally constant, which is a fair assumption. FPS stays pretty constant after the
    start (I should investigate why the FPS does slow down though!)

    :param environment_steps: Total number of timesteps (steps in the environments) (we get this from the printed
    terminal logs (`total_timesteps` variable in SB3)
    :param seconds:  Time in seconds since the beginning of training (`time_elapsed` variable in SB3)
    """
    short_training_run: int = 1e8
    long_training_run: int = 5e8

    elapsed_mins = seconds_to_minutes(seconds)
    elapsed_hours = seconds_to_hours(seconds)

    steps_per_min = environment_steps / elapsed_mins
    steps_per_hour = environment_steps / elapsed_hours

    print(f"\n\nEstimating time for a full training run of {short_training_run} to {long_training_run} timesteps")
    print(f"Time elapsed: {elapsed_mins} minutes ({elapsed_hours} hours)")
    print(f"Environment steps: {environment_steps}")
    print(f"Environment steps per second: {environment_steps / seconds}")
    print(f"Environment steps per minute: {steps_per_min}")
    print(f"Environment steps per hour: {steps_per_hour}")

    print(f"{short_training_run} steps would take: {short_training_run/ steps_per_min} minutes, or {short_training_run/ steps_per_hour} hours")
    print(f"{long_training_run} steps would take: {long_training_run/ steps_per_min} minutes, or {long_training_run/ steps_per_hour} hours")


if __name__ == '__main__':
    estimate_time_for_full_training_runs(environment_steps=17580000, seconds=12780)



