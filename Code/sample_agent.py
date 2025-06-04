import numpy as np
# import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, dim_action: int):
        """
        Initializes the agent with the specified dimensionality of actions.
        """
        self.dim_action: int = dim_action

    def act(self, ob, reward: float, done: bool, vision_on: bool) -> np.ndarray:
        """
        Decides on an action based on the given observation, reward, done status, and vision flag.
        If vision is enabled, the input contains visual data in addition to other sensory data.
        
        Args:
        - ob: Observation input, which is a tuple containing various sensor and vision data.
        - reward: The reward from the environment for the last action.
        - done: A boolean indicating if the episode has ended.
        - vision_on: A boolean flag indicating if vision data is available.
        
        Returns:
        - A random action selected based on the agent's dimensionality.
        """
        if not vision_on:
            # If vision is off, process the basic observation (ob), if needed.
            pass
        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = ob

            """ The code below is for checking the vision input. This is very heavy for real-time Control
                So you may need to remove.
            """
            # print(vision.shape)
            """
            img = np.ndarray((64,64,3))
            for i in range(3):
                img[:, :, i] = 255 - vision[:, i].reshape((64, 64))

            plt.imshow(img, origin='lower')
            plt.draw()
            plt.pause(0.001)
            """

        # Return a random action by generating a random array of the correct size and applying np.tanh for bounded output
        return np.tanh(np.random.randn(self.dim_action))  # random action
