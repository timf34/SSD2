import wandb
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

# Note: this currently doesn't work. Videos sort of get saved, however they don't seem to get saved properly.
# However, there is an easy fix as wandb will save videos by default if you pass `monitor_gym=True` to the wandb.init!
# For example, here is a run that didn't work using the below code: https://wandb.ai/timf34/sb3_train/runs/2f7os9c8/overview
# Here is a run that did work using `monitor_gym=True`: https://wandb.ai/timf34/sb3_train/runs/2lmo0z9j/files?workspace=user-timf34


class WandbVecVideoRecorder(VecVideoRecorder):
    def __init__(self, venv, directory, record_video_trigger, video_length=200, use_wandb=True):
        super(WandbVecVideoRecorder, self).__init__(venv, directory, record_video_trigger, video_length)
        self.use_wandb = use_wandb

    def step_wait(self):
        # Just overwriting the `step_wait` method of the `VecVideoRecorder` class in order to log videos to wandb.
        # Not sure whether its best to use this or `monitor_gym=True` in wandb.init as mentioned above.
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()
                if self.use_wandb:
                    print("saving wandb video: ", self.video_recorder.path)
                    wandb.log({"my video": wandb.Video(self.video_recorder.path)})
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos