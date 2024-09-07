import numpy as np
import matplotlib.pyplot as plt

tex_preamble = r"""
\usepackage{xcolor}
\definecolor{C0}{HTML}{1d1d1d}
\color{C0}

\usepackage[no-math]{fontspec}

\emergencystretch=8pt
\hyphenpenalty=1000 % default 50
\tolerance=800      % default 200
%\righthyphenmin=4
%\lefthyphenmin=4

\setmainfont[
    BoldFont = Vollkorn Bold,
    ItalicFont = Vollkorn Italic,
    BoldItalicFont={Vollkorn Bold Italic},
    RawFeature=+lnum,
]{Vollkorn}

\setsansfont[
    BoldFont = Lato Bold,
    FontFace={l}{n}{*-Light},
    FontFace={l}{it}{*-Light Italic},
]{Lato}

\usepackage{unicode-math}
\mathitalicsmode=1

\defaultfontfeatures{Extension = .otf}
\setmathfont[
mathit = sym,
mathup = sym,
mathbf = sym,
math-style = TeX, 
bold-style = TeX
]{Wholegrain-Math}
\everydisplay{\Umathoperatorsize\displaystyle=4ex}
\AtBeginDocument{\renewcommand\setminus{\smallsetminus}}

"""


#colors = ['#1d1d1d','#1e3264','#bc2435','#709f63','#60b5d6','#ffad5f']

colors = ['#1d1d1d','#1e3264','#c82336','#198c71','#ef9646','#1ca9d2']

markers = ['+','o','x','d','+','^']
m_sizes = [6,2,4,1.8,5,2]

linestyles = ['-','-','--','-.',(0, (3, 1.5, 1, 1, 1, 1.5)),':']

line_styles = \
{
    'l0' : {'lw':.6,'color':colors[0],'marker':markers[0],'ms':m_sizes[0]},
    'l1' : {'lw':.6,'color':colors[1],'marker':markers[1],'ms':m_sizes[1]},
    'l2' : {'lw':.6,'color':colors[2],'marker':markers[2],'ms':m_sizes[2]},
    'l3' : {'lw':.6,'color':colors[3],'marker':markers[3],'ms':m_sizes[3]},
    'l4' : {'lw':.6,'color':colors[4],'marker':markers[4],'ms':m_sizes[4]},
    'l5' : {'lw':.6,'color':colors[5],'marker':markers[5],'ms':m_sizes[5]},
}

line_styles_nomarker = \
{
    'l0' : {'lw':1,'color':colors[0],'ls':linestyles[0]},
    'l1' : {'lw':1,'color':colors[1],'ls':linestyles[1]},
    'l2' : {'lw':1,'color':colors[2],'ls':linestyles[2]},
    'l3' : {'lw':1,'color':colors[3],'ls':linestyles[3]},
    'l4' : {'lw':1,'color':colors[4],'ls':linestyles[4]},
    'l5' : {'lw':1,'color':colors[5],'ls':linestyles[5]},
}

line_styles_nomarker1 = \
{
    'l0' : {'lw':.4,'color':colors[0]},
    'l1' : {'lw':.4,'color':colors[1]},
    'l2' : {'lw':.4,'color':colors[2]},
    'l3' : {'lw':.4,'color':colors[3]},
    'l4' : {'lw':.4,'color':colors[4]},
    'l5' : {'lw':.4,'color':colors[5]},
}


cm = 1/2.54

figure_width = 13.5
image_width = 10
left_right_margin = (figure_width - image_width)/2

image_height = 5.5
top_margin = .5
bottom_margin = 1
inner_margin = .25

#top_bottom_margin = (figure_height - image_height)/2

figure_height = top_margin+bottom_margin+image_height

left   = left_right_margin / figure_width
bottom = bottom_margin / figure_height

width  = image_width/figure_width
height = image_height/figure_height



#height_scale3 = .7
#figure_height3 = top_margin+bottom_margin + 2*inner_margin + 3*height_scale3*image_height

#bottom3 = bottom_margin / figure_height3

#height3 = height_scale3*image_height / figure_height3
#inner_m3 = inner_margin / figure_height3

width2 = (image_width-inner_margin)/(2*figure_width)
hoffset2 = width2+inner_margin/figure_width

width3 = (image_width-2*inner_margin)/(3*figure_width)
hoffset3 = width3+inner_margin/figure_width




def mystep(x,y, ax=None, where='post', **kwargs):
    # https://stackoverflow.com/questions/44961184/matplotlib-plot-only-horizontal-lines-in-step-plot
    assert where in ['post', 'pre']
    x = np.array(x)
    y = np.array(y)
    if where=='post': y_slice = y[:-1]
    if where=='pre': y_slice = y[1:]
    X = np.c_[x[:-1],x[1:],x[1:]]
    Y = np.c_[y_slice, y_slice, np.zeros_like(x[:-1])*np.nan]
    if not ax: ax=plt.gca()
    return ax.plot(X.flatten(), Y.flatten(), **kwargs)