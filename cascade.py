import copy


def _is_branch(mod, img):
    return mod.predict([img.get()]) == 1


def _split_image(img, min_size=200):
    if img.get().shape[0] > min_size and img.get().shape[1] > min_size:
        x = int(img.get().shape[0] / 2)
        y = int(img.get().shape[1] / 2)
        images = []
        for i in range(1, 5):
            im = Image(img.get(), copy.deepcopy(img.coordinates()))
            im.slice(x, y, i)
            images.append(im)
        return images


class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def draw_on(self, img):
        red = (255, 0, 0)
        w = int(max(img.shape[0] * 0.005, 1.))
        img[self.x1:(self.x2 + w), self.y1:(self.y1 + w)] = red
        img[self.x1:(self.x2 + w), self.y2:(self.y2 + w)] = red
        img[self.x1:(self.x1 + w), self.y1:(self.y2 + w)] = red
        img[self.x2:(self.x2 + w), self.y1:(self.y2 + w)] = red

    def __repr__(self):
        return "(%d %d,%d %d)" % (self.x1, self.y1, self.x2, self.y2)


class Image:
    def __init__(self, img, rect=None):
        self._img = img
        if rect is None:
            self._rect = Rect(0, 0, self._img.shape[0] - 1, self._img.shape[1] - 1)
        else:
            self._rect = rect
        
    def get(self):
        return self._img
    
    def slice(self, x, y, quadrant):
        if quadrant == 1:
            self._img = self._img[:x, y:]
            self._rect.x2 -= x
            self._rect.y1 += y
        elif quadrant == 2:
            self._img = self._img[:x, :y]
            self._rect.x2 -= x
            self._rect.y2 -= y
        elif quadrant == 3:
            self._img = self._img[x:, :y]
            self._rect.x1 += x
            self._rect.y2 -= y
        elif quadrant == 4:
            self._img = self._img[x:, y:]
            self._rect.x1 += x
            self._rect.y1 += y
        else:
            raise RuntimeError("no such quadrant")
            
    def coordinates(self):
        return self._rect


def contains_palmtree(mod, img):
    if _is_branch(mod, img):
        return True
    else:
        sp = _split_image(img)
        if sp is None:
            return False
        else:
            return (contains_palmtree(mod, sp[0]) or contains_palmtree(mod, sp[1]) or
                    contains_palmtree(mod, sp[2]) or contains_palmtree(mod, sp[3]))


def palmtree_markers(rects, mod, img):
    if _is_branch(mod, img):
        rects.append(img.coordinates())
        return 
    else:
        sp = _split_image(img)
        if sp is None:
            return
        else:
            for sp_img in sp:
                palmtree_markers(rects, mod, sp_img)


def extract_all_images(images, img, min_size):
    images.append(img)
    sp = _split_image(Image(img), min_size)
    if sp is None:
        return
    else:
        for sp_img in sp:
            extract_all_images(images, sp_img.get(), min_size)
