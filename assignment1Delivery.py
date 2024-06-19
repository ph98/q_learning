import pygame
import gymnasium as gym
import math
import numpy as np

homeImage = pygame.image.load('./delivery-images/home1.png')
restaurantImage = pygame.image.load('./delivery-images/restaurant.png')
treeImage0 = pygame.image.load('./delivery-images/tree.png')
treeImage1 = pygame.image.load('./delivery-images/tree1.png')
treeImage2 = pygame.image.load('./delivery-images/tree2.png')
treeImage3 = pygame.image.load('./delivery-images/tree3.png')
deliveryImage = pygame.image.load('./delivery-images/delivery.png')
deliveryWithBurgurImage = pygame.image.load('./delivery-images/delivery-with-burger.png')
holeImage = pygame.image.load('./delivery-images/hole.png')

navigations = [ (0, 1), (0, -1), (1, 0), (-1, 0) ]

rewards = {
    'restaurant': 50,
    'customer': 200,
    'hole': -100,
    # 'out_of_bounds': -10, # and trees
    'step': -1
}

# wirte a random stuff

class DeliveryEnv(gym.Env):
    """ initialises the delivery environment

    Args:
        no args
    """
    def __init__(self, goal_coordinates = (4,4), restourantPosition= (0, 4), tree_array=[]):
        """ 
            initialises the delivery environment
        """
        super().__init__()

        self.height = 10
        self.width = 10
        self.cell_size = 100
        self.action_space = gym.spaces.Discrete(4)
        self.customerPosition = goal_coordinates
        self.holesArray = []
        self.restourantPosition = restourantPosition
        self.treesArray = tree_array
        
        self.reset()

        # pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        self.clock = pygame.time.Clock()


    def add_hell_states(self, hell_state_coordinates):
        """
        add a hole in the environment
        args:
            hell_state_coordinates: tuple
        """
        self.holesArray.append(hell_state_coordinates)
        pass

    def reset(self):
        """
         reset the environment and return the initial state and distance to goal
        """

        # generate a random position for the delivery agent, it should not be in customer, or trees, or holes
        self.deliveryPosition = (np.random.randint(self.height), np.random.randint(self.width))
        while self.deliveryPosition == self.customerPosition or self.deliveryPosition in self.treesArray or self.deliveryPosition in self.holesArray:
            self.deliveryPosition = (np.random.randint(self.height), np.random.randint(self.width))
        
        self.turn = 0
        self.gotFoodFromRestaurant = (self.deliveryPosition == self.restourantPosition) and True or np.random.choice([True, False])
        # self.gotFoodFromRestaurant = True

        distance_to_goal = abs(self.deliveryPosition[0] - self.customerPosition[0]) + abs(self.deliveryPosition[1] - self.customerPosition[1])
        return (*self.deliveryPosition, self.gotFoodFromRestaurant and 1 or 0), {
            distance_to_goal: distance_to_goal,
            'has_food': self.gotFoodFromRestaurant
        }


    def step(self, action):
        """
        take a step in the environment using an action including moving the delivery agent, checking if the delivery agent is at the restaurant position, customer position, or out of bounds
        args:
            action: int
        returns:
            deliveryPosition: tuple
            reward: float
            done: bool
            info: dict
        """
        # reward is negative for each step
        reward = rewards['step']
        done = False
        self.turn += 1
        # move the delivery agent
        move = navigations[action]
        tempDeliveryPosition = (self.deliveryPosition[0] + move[0], self.deliveryPosition[1] + move[1])


        should_move = True
        # dont move if there is a tree, or out of bounds
        if tempDeliveryPosition in self.treesArray or tempDeliveryPosition[0] < 0 or tempDeliveryPosition[0] >= self.height or tempDeliveryPosition[1] < 0 or tempDeliveryPosition[1] >= self.width:
            # reward += rewards['out_of_bounds']
            should_move = False
        
        self.deliveryPosition = should_move and tempDeliveryPosition or self.deliveryPosition

        # check if the delivery agent is at the restaurant position
        if (self.deliveryPosition == self.restourantPosition) and not self.gotFoodFromRestaurant:
            self.gotFoodFromRestaurant = True
            # print('Got food from restaurant!')
            reward += rewards['restaurant']

        # going to customer without food is our hell state
        if self.deliveryPosition == self.customerPosition and not self.gotFoodFromRestaurant:
            # print('Hell state!')
            reward += rewards['hole']
            done = True

        # going to holes is also a hell state
        if self.deliveryPosition in self.holesArray:
            reward += rewards['hole']
            done = True

        

        # check if the delivery agent is at the customer position
        if self.deliveryPosition == self.customerPosition and self.gotFoodFromRestaurant:
            reward += rewards['customer']
            done = True
        

        info = {
            'distance_to_goal': abs(self.deliveryPosition[0] - self.customerPosition[0]) + abs(self.deliveryPosition[1] - self.customerPosition[1]) 
            if self.gotFoodFromRestaurant 
            else abs(self.deliveryPosition[0] - self.restourantPosition[0]) + abs(self.deliveryPosition[1] - self.restourantPosition[1]) + self.width,
            'has_food': self.gotFoodFromRestaurant
        }

        # print('other', self.deliveryPosition, reward, done)
        return (*self.deliveryPosition, self.gotFoodFromRestaurant and 1 or 0), reward, done, info


    
        
    def _renderPicture(self, image, i, j):
        """
        a helper function to render a picture in the environment using a given png/svg file in the specified position
        args:
            image: pygame.image
            i: int
            j: int
        """
        pygame.draw.rect(self.screen, (255, 255, 255), (j * self.cell_size, i * self.cell_size, self.cell_size - 1, self.cell_size - 1))    
        scaledImage = pygame.transform.scale(image, (self.cell_size, self.cell_size))
        self.screen.blit(scaledImage, (j * self.cell_size, i * self.cell_size))

        pass
    
    def _renderTree(self, i, j, turn=0):
        """
        a helper function to render a tree in the environment
        args:
            i: int
            j: int
            turn: int
        """
        # sorry, I am not good at drawing trees -\_(-_-)_/-
        # but it should feel like it is windy
        treeImages = [treeImage0, treeImage1, treeImage2, treeImage3, treeImage2, treeImage1]
        # it will render a new tree every 30 turns
        randomTree = treeImages[math.floor(turn / 30) % 4]
        gap = self.cell_size / 3
        pygame.draw.rect(self.screen, (255, 255, 255), (j * self.cell_size, i * self.cell_size, self.cell_size - 1, self.cell_size - 1))    
        scaledImage = pygame.transform.scale(randomTree, (self.cell_size - gap, self.cell_size))
        self.screen.blit(scaledImage, (j * self.cell_size + gap / 2, i * self.cell_size))
        pass
    
    def close(self):
        """
        close the environment
        """
        pygame.quit()
        pass
    

    def render(self):
        """
        render the environment, draw the environment, trees, delivery agent, restaurant, customer, and update the screen
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.screen.fill((200, 200, 200))

        # draw the environment
        for i in range(self.height):
            for j in range(self.width):
                color = (255, 255, 255)
                if (i, j) == self.deliveryPosition:
                    if(self.gotFoodFromRestaurant):
                        self._renderPicture(deliveryWithBurgurImage, i, j)
                    else:
                        self._renderPicture(deliveryImage, i, j)
                    continue
                if (i, j) == self.customerPosition:
                    self._renderPicture(homeImage, i, j)
                    continue
                if (i, j) == self.restourantPosition:
                    self._renderPicture(restaurantImage, i, j)
                    continue
                if (i, j) in self.treesArray:
                    self._renderTree(i, j, self.turn)
                    continue
                if (i, j) in self.holesArray:
                    self._renderPicture(holeImage, i, j)
                    continue
                pygame.draw.rect(self.screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size - 1, self.cell_size - 1))    
        

        # update the screen
        pygame.display.flip()
        self.clock.tick(600)

        #show the turn number on the window title and console
        # print(f'Turn: {self.turn}')
        pygame.display.set_caption(f'Turn: {self.turn}')

        pass


        
def create_env(goal_coordinates,
               restourantPosition,
               hell_state_coordinates,
               tree_array):
    # Create the environment:
    # -----------------------
    env = DeliveryEnv(goal_coordinates=goal_coordinates, restourantPosition=restourantPosition, tree_array=tree_array)

    for i in range(len(hell_state_coordinates)):
        env.add_hell_states(hell_state_coordinates=hell_state_coordinates[i])

    return env

if __name__ == "__main__":
    env = create_env(goal_coordinates=(4, 5),
                     restourantPosition=(0, 4),
                     hell_state_coordinates=[(1, 2), (4, 0)],
                     tree_array=[(1, 1), (1, 3), (1, 4), (2, 4)]
            )

    env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
    env.close()
    print("Environment closed.")