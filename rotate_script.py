import os

def main():
    min_angle = -90
    max_angle = +90
    for i in range(min_angle, max_angle, 30):
        for j in range(min_angle, max_angle, 30):
            print("{} {}".format(j, i))
            os.system("./project shading_brick.png {} {}".format(j, i))

main()
