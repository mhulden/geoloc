#!/usr/bin/env python

#####################################################################################
# Simple map plotting of --print-matrix output from geoloc                          #
#                                                                                   #
# Usage: reads geoloc matrix from stdin and produces a plot on map                  #
#                                                                                   #
# Plot matrix on Europe map:                                                        #
# geoloc --classify --print-matrix mymsg.txt | ./geoplot.py --europe                #
#                                                                                   #
# Plot on world map, projection is mercator:                                        #
# geoloc --classify --print-matrix mymsg.txt | ./geoplot.py --world --proj=merc     #
#                                                                                   #
# Plot on U.S. map, use contours instead of grid:                                   #
# geoloc --classify --print-matrix mymsg.txt | ./geoplot.py --us --contour          #
#                                                                                   #
# As previous, but output PDF:                                                      #
# geoloc --classify --print-matrix mymsg.txt | ./geoplot.py --us --contour --pdf    #
#                                                                                   #
# MH20140406                                                                        #
#####################################################################################

import getopt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import tempfile

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


pdf = False;
proj = 'cyl'
maptype = 'world';
grid = 0;
contour = 0;

options, remainder = getopt.gnu_getopt(sys.argv[1:], 'uewp:g:cfgn', ['us','europe','world','proj=','grid=','contour','pdf','png'])

for opt, arg in options:
    if opt in ('-u', '--us'):
        maptype = 'us'
    elif opt in ('-w', '--world'):
        maptype = 'world'
    elif opt in ('-e', '--europe'):
        maptype = 'europe'
    elif opt in ('p', '--proj'):
        proj = arg
    elif opt in ('g', '--grid'):
        grid = int(arg)
    elif opt in ('c', '--contour'):
        contour = 1
    elif opt in ('f', '--pdf'):
        pdf = True
    elif opt in ('n', '--png'):
        png = True

if maptype == 'world':
    (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat) = (-180,-70,180,70)

if maptype == 'europe':
    (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat) = (-30,30,40,70)

if maptype == 'us':
    (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat) = (-130,20,-55,50)

c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('white'), c('white'), 0.1, c('white'), c('yellow'), 0.33, c('yellow'), c('LightGreen'), 0.66, c('LightGreen'), c('orange'), 0.8, c('orange'), c('red'), 0.95, c('red')])

data = np.loadtxt(sys.stdin, delimiter="\t")
longranularity = data.shape[1]
latgranularity = longranularity/2
lcenterskip = (180/longranularity)

m = Basemap(projection=proj,lon_0=0, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='l')

#m.bluemarble()
m.drawcoastlines()
m.drawcountries()
m.drawstates()

if grid > 0:
    parallels = np.arange(-90, 90, grid)
    meridians = np.arange(-180, 180, grid)
    m.drawparallels(parallels)
    m.drawmeridians(meridians)

if contour:
    x = np.linspace(-180 + lcenterskip, 180 - lcenterskip, longranularity)
    y = np.linspace(-90 + lcenterskip, 90 - lcenterskip, latgranularity)
    x, y = np.meshgrid(x, y)
    converted_x, converted_y = m(x, y)
    p = m.contourf(converted_x, converted_y, data, cmap=rvb, zorder=1, alpha=0.8)
    #p = m.contour(converted_x, converted_y, data, cmap=rvb, zorder=1, alpha=0.8)
else:
    x = np.linspace(-180 , 180, longranularity, endpoint=False)
    y = np.linspace(-90 , 90, latgranularity, endpoint=False)
    x, y = np.meshgrid(x, y)
    converted_x, converted_y = m(x, y)
    p = m.pcolormesh(converted_x, converted_y, data, cmap=rvb, zorder=1, alpha=0.8)

m.colorbar(p)

if png:
#    tmpfn = os.tempnam('/tmp/','locpng')
    tmpfn = '/tmp/locpng.png'
    plt.savefig(tmpfn, format='png')

#    if sys.platform.startswith('darwin'):
#        os.system("open "+tmpfn)
#    elif sys.platform.startswith('linux'):
#        os.system("xdg-open "+tempfn)
#    elif sys.platform.startswith('win32'):
#        os.startfile(tmpfn)

elif pdf:
    tmpfn = os.tempnam('/tmp/','locpdf')
    plt.savefig(tmpfn, format='pdf')

    if sys.platform.startswith('darwin'):
        os.system("open "+tmpfn)
    elif sys.platform.startswith('linux'):
        os.system("xdg-open "+tempfn)
    elif sys.platform.startswith('win32'):
        os.startfile(tmpfn)

else:
    plt.show()
