def in_image(point, width, height):
    if width > height:
        factor = height/width
        return point[0] >= -0.5 and point[0] <= 0.5 and \
            point[1] >= factor * -0.5 and point[1] <= factor * 0.5
    # height >= width
    factor = width/height
    return point[1] >= -0.5 and point[1] <= 0.5 and \
        point[0] >= factor * -0.5 and point[0] <= factor * 0.5


        
# def in_image(point, factor):
    # return point[0] >= -0.5 and point[0] <= 0.5 \
        # and point[1] >= factor * -0.5 and point[1] <= factor * 0.5
