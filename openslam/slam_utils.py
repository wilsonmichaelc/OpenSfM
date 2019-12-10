def in_image(point, factor):
    return point[0] >= -0.5 and point[0] <= 0.5 \
        and point[1] >= factor * -0.5 and point[1] <= factor * 0.5
