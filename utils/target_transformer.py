import numpy as np

class Reshape:
    def __init__(self, shape):
        self.shape = shape
    def __call__(self, data):
        return data.view(self.shape)

class ToInt:
    def __call__(self, data):
        return int(data[0])

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, label):
        for t in self.transforms:
            label = t(label)
        return label


class BBtoMASK:
    def __call__(self, size, bb):
        mask = self.generatemask(size, bb)
        local = self.generateLabel(size, bb)
        bbnorm, min_size = self.generateBBnorm(size, bb)
    
        new_label = np.concatenate((mask, local, bbnorm), axis=0).reshape(-1)
        new_label = np.concatenate((np.array([6, size[0], size[1]]), new_label), axis=0)
        new_label = new_label.astype(np.float32)
    
    
        return new_label 


    def generatemask(self, size, label_content):
    
        mask = np.zeros((1,size[0],size[1]))
    
        for label in label_content:
            l,t,r,b = label
            l = int(float(l)+0.5)
            t = int(float(t) + 0.5)
            r = int(float(r) + 0.5)
            b = int(float(b) + 0.5)
    
            l = min(size[1]-1, max(0, l))
            t = min(size[0]-1, max(0, t))
            r = min(size[1]-1, max(0, r))
            b = min(size[0]-1, max(0, b))
    
            mask[0, t:b, l:r] = 1.0
    
    
        return mask
    
    def generateLabel(self, size, label_content):
        height = size[0]
        width = size[1]
    
    
        local = np.zeros((4, height, width))
        tmp_dict1 = {}
        tmp_dict2 = {}
    
        for label in label_content:
            l, t, r, b = label
            l = float(l)
            t = float(t)
            r = float(r)
            b = float(b)
    
            l = min(size[1]-1, max(0, l))
            t = min(size[0]-1, max(0, t))
            r = min(size[1]-1, max(0, r))
            b = min(size[0]-1, max(0, b))
    
            for y in range(int(t),int(b)):
                for x in range(int(l),int(r)):
                    local[0,y,x] = (l-x)/width
                    local[1,y,x] = (t-y)/height
                    local[2,y,x] = (r-l)/width
                    local[3,y,x] = (b-t)/height
    
        return local
    
    def overlap(self, bb1, bb2):
        o_l = max(bb1[0], bb2[0])
        o_r = min(bb1[2], bb2[2])
        if o_r <= o_l:
            return 0,0,0,0
        o_t = max(bb1[1], bb2[1])
        o_b = min(bb1[3], bb2[3])
        if o_b <= o_t:
            return 0,0,0,0
    
        return o_l, o_t, o_r, o_b
    
    def generateDifferenceMask(self, bb, bb_list):
        l,t,r,b = bb
        mask = np.ones((1,b-t,r-l))
    
        for tmp in bb_list:
            if tmp == bb:
                continue
            o_l,o_t,o_r,o_b = self.overlap(bb, tmp)
    
            if o_l*o_t*o_r*o_b != 0:
                mask[0, o_t-t:o_b-t, o_l-l:o_r-l] = 0
    
    
        return mask, np.where(mask == 1)[0].__len__()
    
    #def generateBBnorm(self, size, label_content):
    #
    #    BBnorm = np.zeros((1,size[0],size[1]))
    #    min_size = size[0]*size[1]
    #
    #    bb_list = []
    #    for label in label_content:
    #        l, t, r, b = label
    #        l = int(float(l) + 0.5)
    #        t = int(float(t) + 0.5)
    #        r = int(float(r) + 0.5)
    #        b = int(float(b) + 0.5)
    #
    #        l = min(size[1]-1, max(0, l))
    #        t = min(size[0]-1, max(0, t))
    #        r = min(size[1]-1, max(0, r))
    #        b = min(size[0]-1, max(0, b))
    #
    #        bb_list.append([l,t,r,b])
    #    for bb in bb_list:
    #        difference_mask, pixel_num = self.generateDifferenceMask(bb, bb_list)
    #
    #        ind = np.where(difference_mask!=0)
    #
    #        if pixel_num != 0:
    #            BBnorm[0, ind[1]+bb[1], ind[2]+bb[0]] = 1.0/pixel_num
    #
    #            min_size = min(min_size, pixel_num)
    #        else:
    #            BBnorm[0, ind[1]+bb[1], ind[2]+bb[0]] = 0
    #
    #    return BBnorm, min_size

    def generateBBnorm(self, size, label_content):
    
        BBnorm = np.zeros((1,size[0],size[1]))
        min_size = size[0]*size[1]
    
        bb_list = []
        for label in label_content:
            l, t, r, b = label
            l = int(float(l) + 0.5)
            t = int(float(t) + 0.5)
            r = int(float(r) + 0.5)
            b = int(float(b) + 0.5)
    
            l = min(size[1]-1, max(0, l))
            t = min(size[0]-1, max(0, t))
            r = min(size[1]-1, max(0, r))
            b = min(size[0]-1, max(0, b))
    
            bb_list.append([l,t,r,b])
        for bb in bb_list:
            difference_mask, pixel_num = self.generateDifferenceMask(bb, bb_list)
    
            ind = np.where(difference_mask!=0)
    
            if pixel_num != 0:
                BBnorm[0, ind[1]+bb[1], ind[2]+bb[0]] = 1.0
    
                min_size = min(min_size, pixel_num)
            else:
                BBnorm[0, ind[1]+bb[1], ind[2]+bb[0]] = 0
    
        ind = np.where(BBnorm==1)
        if ind.__len__() != 0 and ind[0].__len__() != 0:
            BBnorm = BBnorm / ind[0].__len__()
        else:
            BBnorm = BBnorm * 0
        return BBnorm, min_size












class BBtoMASK_min:
    def __init__(self, ratio, filter_thre, size=None, bb_min=None, bb_max=None):
        self.ratio = ratio
        self.filter_thre = filter_thre
        self.size = size
        self.bb_min = bb_min
        self.bb_max = bb_max
        if self.size is not None:
            height = size[0]
            width = size[1]
            self.bias_x = np.array([[i for i in range(width)] for j in range(height)])
            self.bias_y = np.array([[i for i in range(height)] for j in range(width)]).T


    def __call__(self, label_content, size=None):
        if self.size is None:
            assert size is not None
        if size is not None:
            assert size is None
        
        if label_content.__len__() == 1:
            label_content = [label_content]
        mask, bb, bbnorm = self.generateBB(size, label_content, self.filter_thre)
        #bbnorm, min_size = self.generateBBnorm(size, label_content, self.filter_thre)
    
        #new_label = np.concatenate((mask, local, bbnorm), axis=0).reshape(-1)
        #new_label = np.concatenate((np.array([6, size[0], size[1]]), new_label), axis=0)
        #new_label = new_label.astype(np.float32)
    
    
        #return new_label 
        return mask, bb, bbnorm


    #def generateMask(self, size, label_content, filter_thre):
    #
    #    mask = np.zeros((1,size[0],size[1]))
    #
    #    for label in label_content:
    #        l,t,r,b,c = label.view(-1)
    #        l = int(float(l)+0.5)
    #        t = int(float(t) + 0.5)
    #        r = int(float(r) + 0.5)
    #        b = int(float(b) + 0.5)
    #
    #        if l < 0 and -1*l > (r-l)*filter_thre:
    #            continue
    #        if t < 0 and -1*t > (b-t)*filter_thre:
    #            continue
    #        if r > size[1] and (r-size[1]) > (r-l)*filter_thre:
    #    	continue
    #        if b > size[0] and (b-size[0]) > (b-t)*filter_thre:
    #    	continue


    #        l = min(size[1]-1, max(0, l))
    #        t = min(size[0]-1, max(0, t))
    #        r = min(size[1]-1, max(0, r))
    #        b = min(size[0]-1, max(0, b))
    #
    #        w = int((r-l)*self.ratio+0.5)
    #        h = int((b-t)*self.ratio+0.5)

    #        mask[0, t+h:b-h, l+w:r-w] = c
    #
    #
    #    return mask
    
    def generateBB(self, size, label_content, filter_thre):
        if self.size is not None:
            height = self.size[0]
            width = self.size[1]
            bias_x = self.bias_x
            bias_y = self.bias_y
            size = self.size
        else:
            height = size[0]
            width = size[1]
            bias_x = np.array([[i for i in range(width)] for j in range(height)])
            bias_y = np.array([[i for i in range(height)] for j in range(width)]).T
    
    
        local = np.zeros((4, height, width))
        mask = np.zeros((1,height,width))
        BBnorm = np.zeros((1,height,width))



    
        for label in label_content:
            l, t, r, b, c = label.view(-1)
            l = float(l)
            t = float(t)
            r = float(r)
            b = float(b)
    
            if l < 0 and -1*l > (r-l)*filter_thre:
                continue
            if t < 0 and -1*t > (b-t)*filter_thre:
                continue
            if r > size[1] and (r-size[1]) > (r-l)*filter_thre:
                continue
            if b > size[0] and (b-size[0]) > (b-t)*filter_thre:
                continue
    
            l = min(size[1]-1, max(0, l))
            t = min(size[0]-1, max(0, t))
            r = min(size[1]-1, max(0, r))
            b = min(size[0]-1, max(0, b))
    
            w = int((r-l)*self.ratio+0.5)
            h = int((b-t)*self.ratio+0.5)
        
            sta_x = int(l+w)#-1
            end_x = int(r-w)#+1
            sta_y = int(t+h)#-1
            end_y = int(b-h)#+1
        
            sign = True
            if self.bb_min is not None:
                sign = sign and (b-t) >= self.bb_min and (r-l) >= self.bb_min
            if self.bb_max is not None:
                sign = sign and (b-t) <= self.bb_max and (r-l) < self.bb_max
        
            if sign:
                local[0,sta_y:end_y,sta_x:end_x] = l - bias_x[sta_y:end_y,sta_x:end_x]
                local[1,sta_y:end_y,sta_x:end_x] = t - bias_y[sta_y:end_y,sta_x:end_x]
                local[2,sta_y:end_y,sta_x:end_x] = r - bias_x[sta_y:end_y,sta_x:end_x]
                local[3,sta_y:end_y,sta_x:end_x] = b - bias_y[sta_y:end_y,sta_x:end_x]
        
                mask[0,sta_y:end_y,sta_x:end_x] = c
        
                if max((end_y-sta_y)*(end_x-sta_x),0) != 0:
                    BBnorm[0,sta_y:end_y,sta_x:end_x] = 1.0/( max((end_y-sta_y)*(end_x-sta_x),0) )


        return mask, local, BBnorm
    
    
    #def generateBBnorm(self, size, label_content, filter_thre):
    #    height = size[0]
    #    width = size[1]
    #
    #
    #    BBnorm = np.zeros((1,size[0],size[1]))
    #   	sum_size = 0
    #
    #    
    #    for label in label_content:
    #        l, t, r, b, c = label.view(-1)
    #        l = float(l)
    #        t = float(t)
    #        r = float(r)
    #        b = float(b)
    #
    #
    #        if l < 0 and -1*l > (r-l)*filter_thre:
    #            continue
    #        if t < 0 and -1*t > (b-t)*filter_thre:
    #            continue
    #        if r > size[1] and (r-size[1]) > (r-l)*filter_thre:
    #    	continue
    #        if b > size[0] and (b-size[0]) > (b-t)*filter_thre:
    #    	continue


    #        l = min(size[1]-1, max(0, l))
    #        t = min(size[0]-1, max(0, t))
    #        r = min(size[1]-1, max(0, r))
    #        b = min(size[0]-1, max(0, b))
    #
    #        w = int((r-l)*self.ratio+0.5)
    #        h = int((b-t)*self.ratio+0.5)

    #        BBnorm[0, int(t+h):int(b-h), int(l+w):int(r-w)] = 1.0/( max((b-t-2*h)*(r-l-2*w),0) )
    #        sum_size += (b-t-2*h)*(r-l-2*w)

    #    #ind = np.where(BBnorm==1)
    #    #if ind.__len__() != 0 and ind[0].__len__() != 0:
    #    #    BBnorm = BBnorm / ind[0].__len__()
    #    #else:
    #    #    BBnorm = BBnorm * 0
    #
    #    return BBnorm,sum_size
    
