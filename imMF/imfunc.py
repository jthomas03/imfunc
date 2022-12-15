# -*- coding: utf-8 -*-
#
# John C. Thomas 2021
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
import numpy.fft as fft
import cv2
import warnings
from scipy.ndimage.filters import convolve
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

class imFunc(object):
    """This class is used to call a variety of image processing definitions.

    """

    def __init__(self):
        """Constructor"""

    def sobel_edge_detection(image, filter, convert_to_degree=False, verbose=False):
        def convolution(image, kernel, average=False):
            image_row, image_col = image.shape
            kernel_row, kernel_col = kernel.shape
            output = np.zeros(image.shape)
            pad_height = int((kernel_row - 1) / 2)
            pad_width = int((kernel_col - 1) / 2)
            padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
            padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
            for row in range(image_row):
                for col in range(image_col):
                    output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                    if average:
                        output[row, col] /= kernel.shape[0] * kernel.shape[1]
            return output

        new_image_x = convolution(image, filter, verbose)
        if verbose:
            plt.imshow(new_image_x, cmap='gray')
            plt.title("Horizontal Edge")
            plt.show()
        new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
        if verbose:
            plt.imshow(new_image_y, cmap='gray')
            plt.title("Vertical Edge")
            plt.show()
        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
        gradient_magnitude *= 255.0 / gradient_magnitude.max()
        if verbose:
            plt.imshow(gradient_magnitude, cmap='gray')
            plt.title("Gradient Magnitude")
            plt.show()
        gradient_direction = np.arctan2(new_image_y, new_image_x)
        if convert_to_degree:
            gradient_direction = np.rad2deg(gradient_direction)
            gradient_direction += 180
        return gradient_magnitude, gradient_direction

    def detect_contour(img, image_shape):
        """
        find contours using cv2
        Args:
            img: np.array()
            image_shape: tuple
        Returns:
            canvas: np.array()
            cnt: list
        """
        canvas = np.zeros(image_shape, np.uint8)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
        plt.title('Largest Contour')
        plt.imshow(canvas)
        plt.show()
        return canvas, cnt
    
    def image_plot(imagein, title='Nanonis Test Image', cbar=1, imsave = 0, loc = None):
        fig, ax = plt.subplots()
        z_min, z_max = imagein.min(), imagein.max()
        x, y = np.meshgrid(np.linspace(1, imagein.shape[1], imagein.shape[1]), np.linspace(1, imagein.shape[0], imagein.shape[0]))
        cout = ax.pcolormesh(x, y, imagein, cmap=None, vmin=z_min, vmax=z_max)
        ax.set_title(title)
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        if cbar == 1:
            fig.colorbar(cout, ax=ax)
        plt.axis('scaled')
        fig.tight_layout()
        if imsave == 1:
            if loc == None:
                print("File location not specified.")
            else:
                fig.savefig(loc, dpi=fig.dpi)
        plt.show()

    def image_fft2(imagein, scaled = 15, pltd = 1, dataout = 0):
        fig, ax = plt.subplots()
        zeropadded = np.array(imagein.shape) * 1
        F2 = fft.fftshift(fft.fft2(imagein, zeropadded)) / imagein.size
        z_min, z_max = abs(F2).min(), (abs(F2).max()/scaled)#abs(F2).max()
        plt.imshow(abs(F2),vmin=z_min, vmax=z_max, cmap='gray')
        if pltd == 1:
            plt.show()
        if dataout == 1:
            return abs(F2)

    def planesubtract(imagein):
        fig, ax = plt.subplots()
        z_min, z_max = np.amin(imagein), np.amax(imagein)
        x, y = np.meshgrid(np.linspace(1, imagein.shape[1], imagein.shape[1]), np.linspace(1, imagein.shape[0], imagein.shape[0]))
        cout = ax.pcolormesh(x, y, imagein, cmap='summer', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(cout, ax=ax)
        print('You will define a triangle, click to begin')
        warnings.filterwarnings("ignore",".*GUI is implemented.*")
        p13 = []
        cnt = 0
        while cnt == 0:
            pts = []
            while len(pts) < 3:
                print('Select 3 corners with mouse')
                pts = np.asarray(plt.ginput(3, timeout=0))
                if len(pts) < 3:
                    print('Too few points, starting over')
                    time.sleep(1)  # Wait a second
            for x, y in pts:
                p13.append([int(round(x,0)-1),int(imagein.shape[0]-round(y,0)),imagein[int(round(x,0)-1)][int(round(imagein.shape[0]-y,0))]])
            cnt = 1
        plt.close()
        p1 = np.array(p13[0])
        p2 = np.array(p13[1])
        p3 = np.array(p13[2])
        v1 = p3 - p1
        v2 = p2 - p1
        cp = np.cross(v1, v2)
        a, b, c = cp
        d = np.dot(cp, p3)
        print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
        planeimg = np.zeros((imagein.shape[0],imagein.shape[1]))
        for i in range(0,planeimg.shape[0]):
            for j in range(0,planeimg.shape[1]):
                planeimg[i][j] = (d - a*i - b*j)/c
        imout = imagein-planeimg
        imot = (imout-(np.amin(imout)))
        fig, ax = plt.subplots()
        z_min, z_max = np.amin(imot), np.amax(imot)
        x, y = np.meshgrid(np.linspace(1, imagein.shape[0], imagein.shape[0]), np.linspace(1, imagein.shape[1], imagein.shape[1]))
        cot = ax.pcolormesh(x, y, imot, cmap='summer', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(cot, ax=ax)
        plt.show() 
        return imot

    def planesubtractpnts(imagein,p13):
        p1 = np.array(p13[0])
        p2 = np.array(p13[1])
        p3 = np.array(p13[2])
        v1 = p3 - p1
        v2 = p2 - p1
        cp = np.cross(v1, v2)
        a, b, c = cp
        d = np.dot(cp, p3)
        print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
        planeimg = np.zeros((imagein.shape[0],imagein.shape[1]))
        for i in range(0,planeimg.shape[0]):
            for j in range(0,planeimg.shape[1]):
                planeimg[i][j] = (d - a*i - b*j)/c
        imout = imagein-planeimg
        imot = (imout-(np.amin(imout)))
        return imot

    
    def imfilter(f,shape=[3,3],sigma=1):
        if f == 'gaussian':
            i, j = [(s-1.)/2 for s in shape]
            y, x = np.ogrid[-i:i+1,-j:j+1]
            h = np.exp(-(x**2 + y**2) / (2.*sigma**2))
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        elif f == 'laplacian':
            h = [[1,1,1],[1,-8,1],[1,1,1]]
            return np.array(h)
        elif f == 'laplacian2':
            h = [[0,1,0],[1,-4,1],[0,1,0]]
            return np.array(h)
        elif f == 'sobel':
            h = [[1,2,1],[0,0,0],[-1,-2,-1]]
            return np.array(h)
        elif f == 'prewitt':
            h = [[1,1,1],[0,0,0],[-1,-1,-1]]
            return np.array(h)
        else:
            return None

    def conv2(x,y,mode='same'):
        if not(mode == 'same'):
            raise Exception("Mode not supported")
        if (len(x.shape) < len(y.shape)):
            dim = x.shape
            for i in range(len(x.shape),len(y.shape)):
                dim = (1,) + dim
            x = x.reshape(dim)
        elif (len(y.shape) < len(x.shape)):
            dim = y.shape
            for i in range(len(y.shape),len(x.shape)):
                dim = (1,) + dim
            y = y.reshape(dim)
        origin = ()
        for i in range(len(x.shape)):
            if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
                x.shape[i] > 1 and
                y.shape[i] > 1):
                origin = origin + (-1,)
            else:
                origin = origin + (0,)
        z = convolve(x,y, mode='constant', origin=origin)
        return z

    def sub_lin_fit(img):
        def bsl_als(y, lam, p, niter=10):
            L = len(y)
            D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
            return z
        out = np.zeros([img.shape[0],img.shape[1]])
        for j in range(0,img.shape[0]):
            tmp = bsl_als(img[j,:], 10**2, 0.001)
            out[j] = img[j,:] - tmp
        return out

    def sub_lin_fit2(img, line):
        def bsl_als(y, lam, p, niter=10):
            L = len(y)
            D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
            return z
        out = np.zeros([img.shape[0],img.shape[1]])
        tmp = bsl_als(img[line,:], 10**2, 0.001)
        for j in range(0,img.shape[0]):
            out[j] = img[j,:] - tmp
        return out

    def unstrip(im, rt, winsize):
        if rt < 0 or rt > 1:
            rt = 0.5
        #take the second order derivative
        B = np.gradient(image)#B = np.diff(im, axis=0)#zeros((im.shape[0],im.shape[1]))
        C = np.gradient(B[0])#C = np.diff(B,axis=0)
        C = C[0]
        D = np.zeros((C.shape[0],C.shape[1]))
        thresh = rt*np.max(C)
        for r in range(0,C.shape[0]):
            for c in range(0,C.shape[1]):
                if abs(C[r][c]) > thresh:
                    D[r][c] = 1
        out = im.copy()
        for r in range(0,D.shape[0]):
            for c in range(0,D.shape[1]):
                if abs(D[r][c]) == 1:
                    if r < (winsize):
                        bot = 0
                        top = r+winsize
                    elif r > (D.shape[0]-winsize-1):
                        bot = r-winsize
                        top = D.shape[0]-1
                    else:
                        bot = r-winsize
                        top = r+winsize
                    out[r][c] = np.median(im[bot:top,c])
        return out, D
    
    def readfile(filein,dtype,direction):
        """
        Reads SXM file and returns array, corrects for scan direction
        """
        def parse_scan_header_table(table_list):
            table_processed = []
            for row in table_list:
                table_processed.append(row.strip('\t').split('\t'))
            keys = table_processed[0]
            values = table_processed[1:]
            zip_vals = zip(*values)
            return dict(zip(keys, zip_vals))
            
        def start_byte(fname):
            with open(fname, 'rb') as f:
                tag = 'SCANIT_END'
                byte_offset = -1
                for line in f:
                    entry = line.strip().decode()
                    if tag in entry:
                        byte_offset = f.tell()
                        break
                if byte_offset == -1:
                    print('SXM file read error')
            return byte_offset

        def read_raw_header(fname,fnamebyt):
            with open(fname, 'rb') as f:
                return f.read(fnamebyt).decode('utf-8', errors='replace')

        def parse_sxm_header(header_raw):
            header_entries = header_raw.split('\n')
            header_entries = header_entries[:-3]
            header_dict = dict()
            entries_to_be_split = ['scan_offset',
                                'scan_pixels',
                                'scan_range',
                                'scan_time']
            entries_to_be_floated = ['scan_offset',
                                    'scan_range',
                                    'scan_time',
                                    'bias',
                                    'acq_time']
            entries_to_be_inted = ['scan_pixels']
            entries_to_be_dict = [':DATA_INFO:']
            for i, entry in enumerate(header_entries):
                if entry in entries_to_be_dict:
                    count = 1
                    for j in range(i+1, len(header_entries)):
                        if header_entries[j].startswith(':'):
                            break
                        if header_entries[j][0] == '\t':
                            count += 1
                    header_dict[entry.strip(':').lower()] = parse_scan_header_table(header_entries[i+1:i+count])
                    continue
                if entry.startswith(':'):
                    header_dict[entry.strip(':').lower()] = header_entries[i+1].strip()
            for key in entries_to_be_split:
                header_dict[key] = header_dict[key].split()
            for key in entries_to_be_floated:
                if isinstance(header_dict[key], list):
                    header_dict[key] = np.asarray(header_dict[key], dtype=np.float)
                else:
                    if header_dict[key] != 'n/a': 
                        header_dict[key] = np.float(header_dict[key])
            for key in entries_to_be_inted:
                header_dict[key] = np.asarray(header_dict[key], dtype=np.int)
            return header_dict

        def load_data(fname,header,byte_offset,dataf,indir):
            channs = list(header['data_info']['Name'])
            nchanns = len(channs)
            nx, ny = header['scan_pixels'] 
            ndir = indir
            data_dict = dict()
            f = open(fname, 'rb')
            byte_offset += 4
            f.seek(byte_offset)
            scandata = np.fromfile(f, dtype='>f4')
            f.close()
            scandata_shaped = scandata.reshape(nchanns, ndir, ny, nx)
            for i, chann in enumerate(channs):
                chann_dict = dict(forward=scandata_shaped[i, 0, :, :],
                                    backward=scandata_shaped[i, 1, :, :])
                data_dict[chann] = chann_dict
            return data_dict
            
        byte = start_byte(filein)
        header = parse_sxm_header(read_raw_header(filein,byte))
        data = dict()
        if 'both' in header['data_info']['Direction']:
            data = load_data(filein,header,byte,'scan',2)
        else:
            data = load_data(filein,header,byte,'scan',1)
        channs = list(header['data_info']['Name'])
        if dtype not in channs:
            dtype = ''
        try:
            if header['scan_dir'] == 'up':
                return data[dtype][direction]
            elif header['scan_dir'] == 'down':
                return np.flipud(data[dtype][direction])
        except:
            print('SXM file read error')

    def otsu(filtered):
        """
        adopted from github.com/mohabmes/Otsu-Thresholding/blob/master/otsu.py
        Apply OTSU threshold
        Args:
            filtered: np.array
        Returns:
            thresh: np.array
        """
        threshold_values = {}
        def Hist(img):
            row, col = img.shape 
            y = np.zeros(256)
            for i in range(0,row):
                for j in range(0,col):
                    tmp = math.floor(img[i,j])
                    y[tmp] += 1
            x = np.arange(0,256)
            plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
            plt.show()
            return y

        def regenerate_img(img, threshold):
            row, col = img.shape 
            y = np.zeros((row, col))
            for i in range(0,row):
                for j in range(0,col):
                    if img[i,j] >= threshold:
                        y[i,j] = 255
                    else:
                        y[i,j] = 0
            return y

        def countPixel(h):
            cnt = 0
            for i in range(0, len(h)):
                if h[i]>0:
                    cnt += h[i]
            return cnt

        def weight(s, e):
            w = 0
            for i in range(s, e):
                w += h[i]
            return w

        def mean(s, e):
            m = 0
            w = weight(s, e)
            for i in range(s, e):
                m += h[i] * i
            return m/float(w)

        def variance(s, e):
            v = 0
            m = mean(s, e)
            w = weight(s, e)
            for i in range(s, e):
                v += ((i - m) **2) * h[i]
            v /= w
            return v  
            
        def threshold(h):
            cnt = countPixel(h)
            for i in range(1, len(h)):
                vb = variance(0, i)
                wb = weight(0, i) / float(cnt)
                mb = mean(0, i)  
                vf = variance(i, len(h))
                wf = weight(i, len(h)) / float(cnt)
                mf = mean(i, len(h))
                V2w = wb * (vb) + wf * (vf)
                V2b = wb * wf * (mb - mf)**2  
                if not math.isnan(V2w):
                    threshold_values[i] = V2w

        def get_optimal_threshold():
            mind = min(threshold_values.values())
            optimal_threshold = [k for k, v in threshold_values.items() if v == mind]
            return optimal_threshold[0]

        h = Hist(filtered)  
        threshold(h)
        op_thres = get_optimal_threshold()
        res = regenerate_img(filtered, op_thres)
        return res

    def readspec(fil):
        data = []
        with open(fil,'r') as f:
            idx = 0
            for i in f:
                tmp = []
                if '[DATA]' in i:
                    idx = 1
                elif idx == 1:
                    idx = 2
                if idx == 2:
                    out = i.split('\t')
                    for j in out:
                        if '\n' in j:
                            sp = j.split('\n')
                            tmp.append(sp[0])
                        else:
                            tmp.append(j)
                    data.append(tmp)
        return data