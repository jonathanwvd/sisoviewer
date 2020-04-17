import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

input_style = {'width': '100%'}
dropdown_style = {'width': '90%', 'float': 'left'}
label_style = {'width': '10%', 'float': 'left', 'font-size': '1.5em'}
div_log = {'max-height': '200px', 'overflow': 'auto', 'display': 'flex', 'flex-direction': 'column-reverse'}


# %% basic functions
def allow_save_to_dataset(full, data_folder):
    save = []
    for d in full:
        hdf_data = pd.HDFStore(data_folder / d['value'])
        if ~ hdf_data['general_info']['read_only'].values[0]:
            save.append(d)
    return save


def fig_template(rows, xaxes_title, yaxes_title):
    """
    The basic template for the figures.

    Parameters
    ----------
    rows : int
        Number of the rows of the figure.
    xaxes_title : string
        Name of the x axis.
    yaxes_title : list
        Names of the y axis.

    Returns
    -------
    fig :
        Figure with basic configurations
    """
    from plotly.subplots import make_subplots

    # figure according to the number of rows
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)

    # default for all figures
    fig.update_layout(height=420, plot_bgcolor='white')
    tick_prop = dict(showline=True,
                     showgrid=False,
                     linecolor='rgb(204, 204, 204)',
                     linewidth=2,
                     ticks='outside')

    fig.update_xaxes(**tick_prop)
    fig.update_yaxes(**tick_prop)

    for r in np.arange(rows):
        # if last row, add x axes
        if r == rows - 1:
            fig.update_xaxes(title_text=xaxes_title, row=r + 1, col=1)
        # else, do not show x axis
        else:
            fig.update_xaxes(showticklabels=False, row=r + 1, col=1)

        fig.update_yaxes(title_text=yaxes_title[r], row=r + 1, col=1)
    return fig


def function_name(keys):
    """
    Gather the functions for each section and generate the dropdown
    Parameters
    ----------
    keys : list
        Dictionary keys of the functions.

    Returns
    -------
        Dropdown options.
    """
    names = [{'label': k, 'value': k} for k in keys]
    return names


# %% layout
def subsection_add_to_plot(prefix, keys, many_variables_allowed):
    """
    Layout for the add to plot subsection
    """
    components = [
        html.H3('Add to plot'),

        html.Div([
            html.Label('Apply', title='select a function'),
            html.Div([
                dcc.Dropdown(id=prefix + '_plot_add_func', options=function_name(keys), style=dropdown_style),
                html.Label('?', id=prefix + '_plot_add_func_info',
                           title='select a function to see the help information', style=label_style),
            ]),
        ], className='input_prop'),
    ]

    if many_variables_allowed:
        components += [
            html.Div([
                html.Label('to variable'),
                dcc.Dropdown(id=prefix + '_plot_add_var')
            ], className='input_prop'),
        ]

    components += [
        html.Div([
            html.Label('with parameters'),
            html.Div(id=prefix + '_plot_add_par', children=[]),
        ], className='input_prop'),

        html.Div([
            html.Label('and name', title='give a name to the new line'),
            dcc.Input(id=prefix + '_plot_add_name', type='text', placeholder='default name', style=input_style),
        ], className='input_prop'),

        html.Button('Add', id=prefix + '_plot_add_bt'),
    ]

    sub = html.Div(components, className='c_l4')
    return sub


def subsection_evaluate(prefix, keys, many_variables_allowed):
    """
    Layout for the evaluate subsection
    """
    components = [
        html.H3('Evaluate from data'),
        html.Div([
            html.Label('Apply', title='select a function'),
            html.Div([
                dcc.Dropdown(id=prefix + '_ev_func', options=function_name(keys), style=dropdown_style),
                html.Label('?', id=prefix + '_ev_func_info', title='select a function to see the help information',
                           style=label_style)
            ]),
        ], className='input_prop'),
    ]

    if many_variables_allowed:
        components += [
            html.Div([
                html.Label('to variable'), dcc.Dropdown(id=prefix + '_ev_var')
            ], className='input_prop'),
        ]

    components += [
        html.Div([
            html.Label('with parameters'),
            html.Div(id=prefix + '_ev_par', children=[]),
        ], className='input_prop'),

        html.Div([
            html.Button('Evaluate', id=prefix + '_ev_bt'),
        ], className='input_prop'),

        html.Div(id=prefix + '_ev_res', children=[], className='border margin', style=div_log),
    ]

    sub = html.Div(components, className='c_l4')
    return sub


def subsection_select_two_variables(prefix, button_name):
    """
    Layout for the selection of two variables
    """
    sub = html.Div([
        html.H3('Select variables'),
        html.Div([
            html.Label('Variable 1'),
            dcc.Dropdown(id=prefix + '_sel1', options=[], value=''),
        ], className='input_prop'),

        html.Div([
            html.Label('Variable 2'),
            dcc.Dropdown(id=prefix + '_sel2', options=[], value=''),
        ], className='input_prop'),

        html.Div([
            html.Button(button_name, id=prefix + '_plot_bt'),
        ], className='input_prop'),

    ],
        className='c_l4',
    )
    return sub


# %% callbacks auxiliary
def time_stamp_to_array(data, measurements):
    min_general = np.min([np.min(data[m].index) for m in measurements])

    for m in measurements:
        # change for seconds
        t = (data[m].index - min_general).view('<i8') / 10 ** 9
        data[m].index = t
    return data


def change_ts(info_loop, ts, measurements, data):
    def ts_variable(data, measurements):
        # interpolates 10000 dots between the range
        max_general = np.min([np.max(data[m].index) for m in measurements])
        min_general = np.max([np.min(data[m].index) for m in measurements])
        Ts = (max_general - min_general) / 1e5
        return Ts

    interpolate = True

    # if Ts is given by the user
    if (ts is not None) and (ts != ''):
        Ts = float(ts)

    # if Ts is variable
    elif info_loop['ts'].values == -3:
        Ts = ts_variable(data, measurements)

    # if Ts is not given by the user but given by the dataset
    elif info_loop['ts'].values > 0:
        Ts = info_loop['ts'].values[0]
        interpolate = False

    # if Ts is given nowhere
    else:
        # Check if Ts is fixed or variable
        measurements_diff1 = np.diff(data[measurements[0]].index, 1)
        measurements_diff2 = np.diff(measurements_diff1, 1)
        Ts = np.mean(measurements_diff1)

        if np.mean(measurements_diff2) / Ts == 0:
            interpolate = False
        else:
            Ts = ts_variable(data, measurements)

    data_interp = {}
    if interpolate:
        max_general = np.min([np.max(data[m].index) for m in measurements])
        min_general = np.max([np.min(data[m].index) for m in measurements])
        t_new = np.arange(min_general, max_general, Ts)

        for m in measurements:
            value_new = np.interp(t_new, data[m].index.values, data[m]['Values'].values)
            data_interp[m] = pd.DataFrame(value_new, t_new, ['Values'])
    else:
        data_interp = data

    return data_interp, Ts


def get_start_and_end(start, end, fig_layout, data_interp, measurements):
    # if start and end are given
    if (start is not None) and (end is not None):
        x0 = start
        x1 = end

    # if relayout
    elif 'xaxis.range[0]' in fig_layout.keys():
        x0 = fig_layout['xaxis.range[0]']
        x1 = fig_layout['xaxis.range[1]']

    # else back to full range
    else:
        x0 = data_interp[measurements[0]].index[0]
        x1 = data_interp[measurements[0]].index[-1]
    return x0, x1


def parameters_update(value, func, prefix, prefix_sub):
    """
    Callback for the update of the parameter inputs according to the selected function
    """
    # if function is not selected
    if value is None:
        raise PreventUpdate

    ch = []
    function_info = func[value]
    info = function_info['description']

    for p in function_info['par'].keys():
        description = function_info['par'][p]['description']
        var_type = function_info['par'][p]['type']
        ch.append(html.Label(p, title=f'{description} ({var_type})'))

        if var_type in ['int', 'float']:
            ch.append(dcc.Input(
                id=f'{prefix}_{prefix_sub}_{p}',
                type='number',
                value=function_info['par'][p]['default'],
                style=input_style
            ))
        else:
            ch.append(dcc.Input(
                id=f'{prefix}_{prefix_sub}_{p}',
                type='text',
                value=function_info['par'][p]['default'],
                style=input_style
            ))
    return ch, info


def get_parameters(par_ch):
    """
    Get parameters from the inputs
    """
    par = []
    for p in par_ch:
        if p['type'] == 'Input':
            if p['props']['type'] == 'text':
                if p['props']['value'][0] == '[':
                    # from input string to list
                    end_list = p['props']['value'][1:-1].split(',')
                    end_list = [float(i) for i in end_list]
                    par.append(end_list)
                else:
                    par.append(p['props']['value'])
            else:
                par.append(p['props']['value'])
    return par


def subsection_evaluate_print(func, var, par_ch, res, module, x, y):
    """
    Build the output of the evaluation subsections
    """
    par = get_parameters(par_ch)

    # apply function
    func_exe = getattr(module, func)
    ev = func_exe(x, y, par)

    # print results
    if isinstance(ev, (int, float)):
        res.append(html.P(f'{func}({var}, {par}) = {ev:.4f}'))
    elif isinstance(ev, list):
        ev_str = [f'{i:.2f}' for i in ev]
        ev_str = ', '.join(ev_str)
        ev_str = '[' + ev_str + ']'
        res.append(html.P(f'{func}({var}, {par}) = ' + ev_str))

    return res


def subsection_add_to_plot_run(func, var, par_ch, name, module, x, y):
    """
    Get the parameters to the new line
    """
    par = get_parameters(par_ch)

    # apply function
    func_exe = getattr(module, func)
    x_add, y_add = func_exe(x, y, par)

    # if name is not given
    if name is None:
        if var == '':
            name = func
        else:
            name = func + '_' + var

    return x_add, y_add, name
