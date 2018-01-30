import pygame, sys
from dm_control import suite

env = suite.load(domain_name="cartpole", task_name="swingup")

pygame.init()
size = width, height = 480, 480

screen = pygame.display.set_mode(size)

counter = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    counter += 1
    env.step([0.5])
    im = env.physics.render(width=width, height=height, camera_id=1).transpose((1,0,2))

    pygame.pixelcopy.array_to_surface(screen, im)
    print(counter)
    pygame.display.update()
    
    