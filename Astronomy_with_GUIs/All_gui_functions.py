from ipywidgets import interactive, widgets, fixed, Layout, FloatSlider, IntSlider
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from wotan import flatten
from transitleastsquares import resample
import pandas as pd
from astropy.timeseries import BoxLeastSquares
import glob
import batman
# Load the data and bin down (too make this demo fast)

lst_alphabet = ['A', 'B', 'C', 'D', 'E', 'F','G','H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
subset_1 = lst_alphabet[:5]
subset_2 = lst_alphabet[5:10]
subset_3 = lst_alphabet[10:15]
lst_subsets = [[' '], subset_1, subset_2, subset_3]
style = {'description_width': 'initial'}


def reading_lightcurve_data(letter):
    file_name = glob.glob('lc_'+letter+'*')
    time, flux = [], []
    print(file_name)
    for x in file_name: 
        print('wtf', x)
        lc_df = pd.read_csv(x)
        time, flux = lc_df['TIME'], lc_df['APER_FLUX']
    return time, flux
    
    
def displaying_plots(letters_list):
    if len(letters_list)>1:
        file_names = 'lc_'+np.array(letters_list)+'.csv'
    
        fig, axes = plt.subplots(5, 1, figsize = (12,12))
        for iii in range(5):
            lc_df = pd.read_csv(file_names[iii])
            time, flux = lc_df['TIME'], lc_df['APER_FLUX']
            
            axes[iii].scatter(time, flux, color = 'C'+str(lst_alphabet.index(letters_list[iii])), s = 10, label = letters_list[iii])
            axes[iii].legend(loc = 2, fontsize = 20, handlelength=0, markerscale = 0., markerfirst = False, handletextpad = 0)
    else: 
        fig, ax = plt.subplots(figsize=(2, 1)) # Adjust figsize as needed
    
        # Hide the axes
        ax.set_axis_off()
        
        # Add text to the figure, centered in the figure's coordinate system
        # x and y are normalized coordinates (0 to 1)
        fig.text(0.5, 0.5, 'Light curves incoming', 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 fontsize= 12, 
                 color='blue')
        
        # Display the figure
    plt.close(fig)

    plt.show()
    return fig


    
def pick_your_lightcurves_gui():
    print("Hello there! Lets start by selecting the set of light curves you want to work with. There are 3 sets, labeled numerically. \n Each contains 5 light curves, 2 or 3 of which include transiting planet signals. \n Your mission: find out which light curve includes a transiting planet signal, and then try and fit that transit so we can learn about the planet!")
    drop_down = widgets.Dropdown(
        options=[(' ', 0), ('One', 1), ('Two', 2), ('Three', 3)],
        value=0,
        description='Select your light curve adventure:',
    )

    init = drop_down.value
#     cityW = widgets.Select(options=geo[init])
    buttons = widgets.RadioButtons(
        options=lst_subsets[init],
        # layout={'width': 'max-content'}, # If the items' names are long
        description='Lets investigate light curves:',
        disabled=True, )
    def creating_buttons_lc_set(indx):
        buttons.options = lst_subsets[indx]
        if indx>0:
            buttons.disabled = False
            output = widgets.Output()
            with output:  # Direct output to the 'output' widget
                # clear_output() # Clear previous output in the output widget
                display(displaying_plots(lst_subsets[indx]))
            container = widgets.HBox([buttons, output])
            display(container)
        
             

        # return displaying_plots(lst_subsets[indx])

    

    # j = widgets.interactive(print_city, city=cityW)
    time_flux = widgets.interactive(reading_lightcurve_data, letter = buttons)

    # i = widgets.interactive(select_city, country=drop_down)
    create_fig = interactive(creating_buttons_lc_set, indx=drop_down)#  
    
    display(create_fig)
    # display(drop_down)
    return time_flux

# Detrending function
def detrending_func(time, flux, method, window_length, break_tolerance, edge_cutoff='0.1', cval=5):
    
    f, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
    if method == 'trim_mean' or method == 'winsorize':
        cval /= 10  # must be a fraction >0, <0.5
        if cval >=0.5:
            cval = 0.49
    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method=method,
        window_length=window_length,
        edge_cutoff=edge_cutoff,
        break_tolerance=break_tolerance,
        return_trend=True,
        cval=cval
        )

    ax[0].plot(time, trend_lc, color='k', linewidth=3)
    ax[0].scatter(time, flux, edgecolors='k', c='yellow', s=30, linewidth=0.5)
    ax[0].set_xlim(min(time), max(time))
    ax[0].set_ylabel('Raw flux')
    ax[1].scatter(time, flatten_lc, edgecolors='k', c='black', s=30)
    ymin, ymax = ax[1].get_ylim()
    ax[1].set_ylim(max(ymin, 0.95), min(1.01, np.median(flatten_lc)+.5*np.std(flatten_lc)))
    ax[1].set_ylabel('Detrended flux')
    plt.xlabel('Time (days)')
    f.subplots_adjust(hspace=0)
    plt.show();

    return time, flatten_lc

def playing_with_detrending_gui(time, flux):

    print('Can you get rid of some noise in the light curve to make the transits clearer? Try it out!')
    try:
        y1=interactive(
            detrending_func,
            time =fixed(time), 
            flux = fixed(flux),
            method=["biweight", "hodges", "welsch", "median", "andrewsinewave", "mean", "trim_mean", "winsorize"],
            window_length=FloatSlider(value=1.,min=0.1,  max=2., step=0.1, style = style, layout = Layout(width='1000px')),
            break_tolerance=FloatSlider(value=0.,min=0.,  max=1., step=0.1, style = style, layout = Layout(width='1000px')),
            edge_cutoff=FloatSlider(value=0.1 ,min=0.,  max=1., step=0.1, style = style, layout = Layout(width='1000px')),
            cval=IntSlider(value=5 ,min=1,  max=9, step=1, style = style, layout = Layout(width='1000px')),
            )
        container = widgets.VBox([widgets.HBox([y1])])
        display(container)
    except:
        print('Sorry, you still need to finish the cell above to run this!')
    return y1
    


def create_fig_plotting_mid_transit_times(time, flux, t0_val, period = 1E3, ax = False):
    time_diff = max(time)-min(time)

    if ax:
        ax.scatter(time, flux, color = 'k', zorder = 10)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.vlines(t0_val, ymin, ymax, lw = 2, color = 'r', zorder = -10)
        if period<time_diff:
            num_per = int(np.ceil(time_diff/period))
            t0_vals = t0_val+np.arange(-1*num_per*10, 10*num_per)*period
            ax.vlines(t0_vals, ymin, ymax, lw = 1, color = 'r', zorder = -100, alpha = 0.7)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.grid(True, axis='x')

    return ax

def create_phase_fold_fig(time, phase_flux, ax = False):
    if ax:
        ax.scatter(time, phase_flux, color = 'k', zorder = 10, alpha =0.5)
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, lw = 2, color = 'r', zorder = -10, alpha = 0.75)
        xmin, xmax = ax.get_xlim()
        lim_val = max(np.abs(xmin), np.abs(xmax))
        ax.set_xlim(-1*lim_val, lim_val)
        ax.set_ylim(ymin, ymax)

    return ax

    
def take_lc_print_phase_space(time, flux, t0_val, period):
    fig, (ax1, ax2)= plt.subplots(1, 2, width_ratios=[1.5, 7],figsize = (20,6), tight_layout = True)
    
    time_diff = max(time)-min(time)

    create_fig_plotting_mid_transit_times(time, flux, t0_val, min(time_diff, period), ax2)
    
    x = ((time - t0_val + 0.5*period) % period) -( 0.5*period)
    m = np.abs(x) < 0.45


    create_phase_fold_fig(x[m], flux[m], ax1)
    plt.show()

    return t0_val, period



def finding_all_transits_gui(time, flux):

    print('Now lets try to find the transits! This time, you can type in what you think is the time of the transit, an then try to find the period by matching the other red llines to other transits. \n Try it out!')

    time = time[~np.isnan(flux)]
    flux = flux[~np.isnan(flux)]

    inter_p = interactive(take_lc_print_phase_space, time = fixed(time), flux = fixed(flux),
                                   t0_val =widgets.FloatText(value = np.median(time),step = 0.05, description='t0:',disabled=False), period = widgets.FloatSlider(value = 3.95, min = 0.5, max = 28., step = 0.01, layout = Layout(width='750px')))
    output = inter_p.children[-1]
    
    output.layout.object_position  = '{horz} {vert}'.format(horz='left', vert='center')
    
    output.layout.height = '500px'
    
    display(inter_p)
    
    return inter_p


def running_median(data, kernel=25):
    """Returns sliding median of width 'kernel' and same length as data """
    
#     print('kernel', kernel)
    
    idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
    idx = idx.astype(np.int64)  # needed if oversamplinfg_factor is not int
    med = np.median(data[idx], axis=1)

    # Append the first/last value at the beginning/end to match the length of
    # data and returned median
#     print('length of med (if 0, need to return 0)', len(med))
    if len(med)>0:
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        med[np.isinf(np.abs(med))] = 0

        return med
    else:
        return np.zeros(len(data))


def showing_BLS_for_planet(time, flux):

    print('Now, you can double heck if you found the right period! Use the code below to see if the strongest signal it finds is the same as your period and if it finds a similar transit time. \n If the periods are different dont worry! Just talk to me, and maybe run the last cell :)')
    time_new = time[~np.isnan(flux)]
    flux_new = flux[~np.isnan(flux)]

    durations = np.linspace(0.01, 0.5, 100)
    model     = BoxLeastSquares(time_new, flux_new)
    max_per = np.min([100., (max(time_new)-min(time_new))*2/3])
    
    results   = model.autopower(durations, frequency_factor = 10, maximum_period=max_per)#, objective='snr', )
    
    print('time len: ', len(time))
    my_median = running_median(results.power, kernel = min((25, int(len(time_new)/10))))
    results['power_final'] = results.power - my_median
#         print('checking median', my_median, set(my_median))
#         print('checking results.power', results.power)
    
    
    
    index = np.argmax(results.power_final)
    period = results.period[index]

    
    t0 = results.transit_time[index]
    duration = results.duration[index]

    plt.figure(figsize = (10, 6))
    val_triangles = min(results.power_final)-np.std(results.power_final)
    ax = plt.gca()
    ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

    plt.xlim(np.min(results.period), np.max(results.period))
    for n in range(2, 10):
        ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
        ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
    plt.ylabel(r'SDE')#, fontsize = 40)
    plt.xlabel('Period (days)')#, fontsize = 40)

    ax.plot(results.period, results.power_final, color = 'k', lw=0.65)
    print('Period: ', period)
    print('t0: ', t0)

    plt.show()
    plt.close()
    if duration<period:
        plt.figure(figsize = (5, 5))
        ax2 = plt.gca()

        x = ((time - t0 + 0.5*period) % period) -( 0.5*period)
        m = np.abs(x) < 0.5
        ax2.scatter(
            x[m],
            flux[m],
            color='k',
            s=5,
            alpha=0.8,
            zorder=10)

        x_new = np.linspace(-0.5, 0.5, 1000)

        f = model.model(x_new + t0, period, duration, t0)

        ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#             ax2.set_xlim(-0.5, 0.5)
        ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
        ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
        plt.show()
    return t0, period
#             print('T0: ', results.transit_time[index], 'duration: ', results.duration[index], 'npoints_dur: ', np.ceil(results.duration[index]/np.nanmedian(np.diff(time))))

def predict_lc(time_lc,t0,P,rp_rs,cosi,a):
    oversample = 4
    e = 0.
    omega = np.pi/2.
    inc = np.arccos(cosi)*180./np.pi
    params = batman.TransitParams()
    params.t0  = t0
    params.per = P
    params.rp  = rp_rs
    params.a   = a
    params.inc = inc
    params.ecc = e
    params.w = omega*180./np.pi
    params.u = []
    params.limb_dark = "uniform"
    print('exp time ', min(np.diff(time_lc)))
        
    m = batman.TransitModel(params, time_lc ,supersample_factor = oversample, exp_time = abs(min(np.diff(time_lc))))

    flux_theo = m.light_curve(params)

    return flux_theo


def creating_transit_model_gui(time, flux, t0, per, rp_rs, a_rs, inc):


    time, flux = np.array(time), np.array(flux)
    time = time[~np.isnan(flux)]
    flux = flux[~np.isnan(flux)]

    model_time = np.linspace(max(time), min(time), len(time))
    model_flux = predict_lc(model_time,t0, per, rp_rs/1E3, np.cos(inc*np.pi/180), a_rs)

    fig, (ax1, ax2)= plt.subplots(1, 2, width_ratios=[1.5, 7],figsize = (20,6), tight_layout = True)
    
    time_diff = max(time)-min(time)

    create_fig_plotting_mid_transit_times(time, flux, t0, min(time_diff, per), ax2)
    ax2.plot(model_time, model_flux, color = 'deepskyblue', zorder = 12, alpha = 0.75, lw = 2.25)
    
    x = ((time - t0 + 0.5*per) % per) -( 0.5*per)
    m = np.abs(x) < 0.35

    model_x = np.linspace(-0.5, 0.5, 200)
    model_f = predict_lc(model_x,0, per, rp_rs/1E3, np.cos(inc*np.pi/180), a_rs)

    create_phase_fold_fig(x[m], flux[m], ax1)
    # xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()

    ax1.plot(model_x, model_f, color = 'deepskyblue', zorder = 12, alpha = 0.95, lw = 2.5)

    ax1.set_xlim(-0.35, 0.35)
    ax1.set_ylim(ymin, ymax)

    plt.show()
    return t0, per, rp_rs, a_rs, inc
    
# 1659.6
    
def modeling_transits_gui(time, flux, t0_val, period):
    print('Great! Now, can you try to get a model to fit the transits? \n Feel free to change the t0 and period, and adjust the other parameters to see how well you can fit the model to the transits. \n Once you do, write it down! At the end of the workshop we can look to see if it agrees with what other people get and the true parameters of the planet!')

    
    interactive_plot = interactive(creating_transit_model_gui, time = fixed(time), flux = fixed(flux),
                                   t0 =widgets.FloatText(value=t0_val,step = 0.01, description='t0:',disabled=False), per = widgets.FloatText(value=period,step = 0.01, description='period:',disabled=False), 
                                   rp_rs = widgets.FloatSlider(value = 50, min = 0., max = 500., step = 0.5, description='planet radius (in star radii * 1000):', layout = Layout(width='750px'), style = style),
                                   a_rs  = widgets.FloatSlider(value =10., min = 0., max = 100., step = 0.05, description='semi-major axis (in star radii):', layout = Layout(width='750px'), style = style),
                                   inc   = widgets.FloatSlider(value = 90., min = 80., max = 90., step = 0.05, description='inclination: ', layout = Layout(width='750px')),
                                  )
    output = interactive_plot.children[-1]
    
    
    output.layout.height = '500px'

    display(interactive_plot)
    return interactive_plot
    
