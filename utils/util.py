import sys
import torch
class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

class feat2labelMap(object):
    def __call__(self, src, labelNumber):
        n,c,h,w = src.size()
        assert c % labelNumber == 0
        ds = int((c/labelNumber)**0.5)
        assert ds*ds == c/labelNumber
        #print '--------a----------'
        #print src
        des = src.view(n, labelNumber, -1) #b
        #print '--------b----------'
        #print des
        des = des.view(n, labelNumber, h*ds, w*ds)#c
        #print '--------c----------'
        #print des
        des = des.transpose(3,2).contiguous()#d
        #print '--------d----------'
        #print des
        des = des.view(n, labelNumber, w*ds, h, ds)#e
        #print '--------e----------'
        #print des
        des = des.transpose(2,3).contiguous()#e
        #print '--------f----------'
        #print des
        des = des.view(n, labelNumber,ds, h, w*ds)#g
        #print '--------g----------'
        #print des
        des = des.transpose(2,3).contiguous()#h
        #print '--------h----------'
        #print des
        des = des.view(n, labelNumber, ds*h, ds*w)#i
        #print '--------i----------'
        #print des

        return des


class Tilling(object):
    def __call__(self, src, labelNumber):
        input_numbers_,input_channels_,input_height_,input_width_ = src.size()
        assert input_channels_ % labelNumber == 0
        ds = int((input_channels_/labelNumber)**0.5)
        assert ds*ds == input_channels_/labelNumber

        tile_dim_ = ds
        tile_dim_sq_ = ds*ds

        output_channels_ = input_channels_ / tile_dim_sq_
        output_width_ = input_width_ * tile_dim_
        output_height_ = input_height_ * tile_dim_
        count_per_output_map_ = output_width_ * output_height_
        count_per_input_map_ = input_width_ * input_height_
        assert 0 == input_channels_ % tile_dim_sq_
        top = torch.Tensor(input_numbers_, input_channels_ / tile_dim_sq_,
            input_height_ * tile_dim_, input_width_ * tile_dim_).cuda()
        top = top.view(-1)
        src = src.view(-1)


        bias = 0
        for n in range(input_numbers_):
          for c in range(output_channels_):
            iy = 0
            oy = 0
            while iy < input_height_:
              ix = 0
              ox = 0
              while ix < input_width_:
                input_channel_offset = 0
                for ty in range(tile_dim_):
                  tx = 0
                  while tx < tile_dim_:
                    top[bias + (oy + ty) * output_width_ + ox + tx] = src.data[bias + input_channel_offset + iy * input_width_ + ix]
                    tx += 1
                    input_channel_offset += count_per_input_map_
                ix += 1
                ox += tile_dim_
              iy += 1
              oy += tile_dim_
            bias += count_per_output_map_
        return torch.autograd.Variable(top.view(input_numbers_, input_channels_ / tile_dim_sq_,
           input_height_ * tile_dim_, input_width_ * tile_dim_))

