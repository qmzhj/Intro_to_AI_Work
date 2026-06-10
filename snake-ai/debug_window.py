"""Pygame 窗口测试：逐个排除问题"""
import pygame
import sys

WIDTH, HEIGHT = 1200, 700

pygame.init()
print("1. pygame.init() OK")

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pygame 测试窗口')
print("2. set_mode OK")

screen.fill((30, 30, 50))
pygame.display.flip()
print("3. 首次 flip OK")
print("4. 窗口应该已经出现了，能否正常交互？")

clock = pygame.time.Clock()
frame = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                print(f"  空格键响应正常 (frame={frame})")

    # 每 60 帧切换颜色，确认画面在更新
    if frame % 60 == 0:
        color = ((frame // 60) * 30 % 200 + 30, 40, 80)
        screen.fill(color)
        pygame.display.flip()

    clock.tick(30)
    frame += 1

    # 避免无响应，定期确认
    if frame % 180 == 0:
        print(f"  运行中... frame={frame}")

pygame.quit()
print("5. 退出正常")
