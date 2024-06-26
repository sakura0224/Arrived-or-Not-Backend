from rmn import RMN


def fer(image):
    m = RMN()
    results = m.detect_emotion_for_single_frame(image)
    image = m.draw(image, results)
    return image
