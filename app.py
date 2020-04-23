# -*- coding: utf-8 -*-
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
from pathlib import Path
import copy

# import modules created for the tool
import modules

# %% basic initial information
log_disabled = True
project_folder = Path.cwd()
map_codes = {'-1': 'unknown', '-2': 'many', '-3': 'variable', '-4': 'anonymous'}
loop_info_print = ['short description', 'industry', 'company', 'type of measurement', 'ts', 'integrating', 'normalised',
                   'year of origin', 'contributor', 'description']

fig_config = {'displaylogo': False,
              'modeBarButtons': [['zoom2d', 'pan2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'autoScale2d',
                                  'resetScale2d', 'toggleSpikelines', 'toImage', 'sendDataToCloud']]}

# find all datasets
data_folder = project_folder / 'data'
datasets = [f.name for f in list(data_folder.glob('*.h5'))]

# dropdown to select the dataset
ld_datasets = [{'label': i.split('.')[0], 'value': i} for i in datasets]
dp_allowed_save_datasets = modules.basic_functions.allow_save_to_dataset(ld_datasets, data_folder)

# globals
hdf_data, info, info_loop, info_general, measurements, Ts, data_interp, data_processed, selected_loop = [None] * 9
data_range, x_start, x_end = [None] * 3
dp_click_plot, dp_click_rg, dp_click_ap, td_click_add, td_click_plot, fd_click_add, fd_click_plot = [0] * 7
cp_click_add, cp_click_plot, pp_click_add, pp_click_plot = [0] * 4

general_log, general_log_collect = [[], []]

# function options for the data processing section
dp_plot_apply_opt = modules.basic_functions.function_name(modules.data_processing.functions.keys())


def to_log(message=None, default=None, par=None):
    global general_log, general_log_collect

    if default is None:
        general_log_collect.append(message)
    else:
        auto_messages = {
            'plotting': f'Plotting {par}...',
            'sel_function': f'Select function first',
            'sel_variable': f'Select variable first',
            'adding_trace': f'Adding trace to {par} plot...',
            'running_ev': f'Running evaluation to {par}',
            'sel_variables': f'Select two variable first',
        }

        general_log_collect.append(auto_messages[default])


tab_style = {
    'padding': '6px',
    'backgroundColor': '#FFFFFF',
}

tab_selected_style = {
    'backgroundColor': '#FFFFFF',
    'padding': '6px'
}

input_style = {'width': '100%'}
dropdown_style = {'width': '90%', 'float': 'left'}
label_style = {'width': '10%', 'float': 'left', 'font-size': '1.5em'}

app = dash.Dash(__name__)
server = app.server

app.title = 'SISO Viewer'

# %% Layout
app.layout = html.Div([

    # first row
    html.Div([

        # %% Title
        html.Div([
            html.H1(
                'SISO Viewer'
            ),

            html.Div([
                html.P(['SISO Viewer is a tool for SISO control loop data visualization and analysis. ',
                        'To get more information about the tool, please check the ',
                        html.A('SISO Viewer page', href='https://www.ufrgs.br/gimscop/repository/siso-viewer/',
                               target='_blank'), '. ', html.Br(),

                        'The project is hosted on ',
                        html.A('GitHub', href='https://github.com/jonathanwvd/sisoviewer', target='_blank'), '. ',
                        "Suggestion or bug report can be sent through this page", html.Br(), html.Br(),

                        html.A('citation', href='https://www.ufrgs.br/gimscop/wp-content/uploads/2020/03/citation.html',
                               target='_blank'), html.Br(),
                        html.A('about', href='https://www.ufrgs.br/gimscop/wp-content/uploads/2020/03/about.html',
                               target='_blank'),
                        ]),
            ],
            ),
        ],
            className='border margin',
            id="c_title",
        ),

        # %% Load data
        html.Div([
            html.H2('Load data'),

            html.Div([
                html.Label('Select dataset:'),
                dcc.Dropdown(id='ld_dataset', options=ld_datasets),
            ],
                className='c_ld_sel',
                id='c_ld_sel_ds'
            ),

            html.Div([
                html.Label('Select loop:'),
                dcc.Dropdown(id='ld_loop'),
            ],
                className='c_ld_sel'
            ),

            html.Div(id='ld_log', children=[], className='border log_opt'),
        ],
            className="border margin",
            id='c_ld'
        ),

        # %% general log
        html.Div([
            html.H2('Log'),

            html.Div(id='gl_log', children=[], className='border log_opt'),
            dcc.Interval(
                id='lg_interval',
                interval=500,  # update log each 0.5 second
                disabled=log_disabled,
            ),

        ],
            className="border margin",
            id='c_gl'
        ),

    ],
        id='c_first_row'
    ),

    html.Div([
        dcc.Tabs([
            # %% Data processing
            dcc.Tab(label='Data processing', style=tab_style, selected_style=tab_selected_style, children=[

                html.Div([
                    # Set sampling time subsection
                    html.Div([
                        html.H3('Set sampling time'),

                        html.Div([
                            html.Label('Sampling time (Ts)', title='Sampling time of the processed time series'),
                            dcc.Input(id='dp_ts', type='text', placeholder='Default Ts', style=input_style),
                        ], className='input_prop'),

                        html.Div([
                            html.Button('Load and plot original data', id='dp_bt_orig'),
                        ], className='input_prop'),

                    ],
                        className='c_l4',
                    ),

                    # Set range subsection
                    html.Div([
                        html.H3('Set range'),
                        html.Div([
                            html.Label('Start', title='First point to be plotted'),
                            dcc.Input(id='dp_plot_rg_start', type='number', placeholder='plot-based', value=None,
                                      style=input_style),
                        ], className='input_prop'),

                        html.Div([
                            html.Label('End', title='Last point to be plotted'),
                            dcc.Input(id='dp_plot_rg_end', type='number', placeholder='plot-based', value=None,
                                      style=input_style),
                        ], className='input_prop'),

                        html.Div([
                            html.Button('Set range', id='dp_bt_rg'),
                        ], className='input_prop'),

                    ],
                        className='c_l4',
                    ),

                    # Apply data processing
                    html.Div([
                        html.H3('Apply data processing'),
                        html.Div([
                            html.Label('Apply', title='select a function'),
                            html.Div([
                                dcc.Dropdown(id='dp_plot_proc', options=dp_plot_apply_opt, style=dropdown_style),
                                html.Label('?', id='dp_plot_proc_info', title='select a function to see the help '
                                                                              'information', style=label_style),
                            ]),
                        ], className='input_prop'),

                        html.Div([
                            html.Label('to variable'),
                            dcc.Dropdown(id='dp_plot_proc_var'),
                        ], className='input_prop'),

                        html.Div([
                            html.Label('with parameters'),
                            html.Div(id='dp_plot_proc_par', children=[]),
                        ], className='input_prop'),

                        html.Div([
                            html.Button('Apply', id='dp_bt_ap'),
                        ], className='input_prop'),

                    ],
                        className='c_l4',
                    ),

                    # Save processed data
                    html.Div([
                        html.H3('Save selected data'),

                        html.Div([
                            html.Label('Select dataset:'),
                            dcc.Dropdown(id='dp_save_dataset', options=dp_allowed_save_datasets),
                        ], className='input_prop'),

                        html.Div([
                            html.Label('Loop name:'),
                            dcc.Input(id='dp_save_name', type='text', placeholder='default is loop name + "_ìndex"',
                                      style=input_style),
                        ], className='input_prop'),

                        html.Div([
                            html.Label('Description'),
                            dcc.Input(id='dp_save_description', type='text', placeholder='default is ""',
                                      style=input_style),
                        ], className='input_prop'),

                        html.Div([
                            html.Label('Short description'),
                            dcc.Input(id='dp_save_short_description', type='text', placeholder='default is ""',
                                      style=input_style),
                        ], className='input_prop'),

                        html.Div([
                            html.Button('Save', id='dp_save_bt'),
                            html.Label(id='dp_save_nooutput'),
                        ], className='input_prop'),

                    ],
                        className='c_l4',
                    ),

                ],
                    className='c_l3',
                ),

                # Plot
                html.Div([
                    dcc.Graph(id='dp_plot', config=fig_config),
                ],
                    className='c_plot',
                ),
            ]),

            # %% Time domain
            dcc.Tab(label='Time domain', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div([
                    html.Div([
                        html.Button('Plot/clean', id='td_plot_bt'),
                    ], className='input_prop'),

                    # add to plot section
                    modules.basic_functions.subsection_add_to_plot('td', modules.time_domain.add.functions.keys(),
                                                                   True),

                    # Evaluation section
                    modules.basic_functions.subsection_evaluate('td', modules.time_domain.evaluate.functions.keys(),
                                                                True),
                ],
                    className='c_l3',
                ),
                # Plot
                html.Div([
                    dcc.Graph(id='td_plot', config=fig_config),
                ],
                    className='c_plot',
                ),
            ]),

            # %% Frequency domain
            dcc.Tab(label='Frequency domain', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div([
                    html.Div([
                        html.Button('Plot/clean', id='fd_plot_bt'),
                    ], className='input_prop'),

                    # add to plot section
                    modules.basic_functions.subsection_add_to_plot('fd', modules.frequency_domain.add.functions.keys(),
                                                                   True),

                    # Evaluation section
                    modules.basic_functions.subsection_evaluate('fd',
                                                                modules.frequency_domain.evaluate.functions.keys(),
                                                                True),
                ],
                    className='c_l3',
                ),
                # Plot
                html.Div([
                    dcc.Graph(id='fd_plot', config=fig_config),
                ],
                    className='c_plot',
                ),
            ]),

            # %% Correlation plot
            dcc.Tab(label='Correlation', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div([
                    # Select variables
                    modules.basic_functions.subsection_select_two_variables('cp', 'Plot correlation'),

                    # add to plot section
                    modules.basic_functions.subsection_add_to_plot('cp', modules.correlation.add.functions.keys(),
                                                                   False),

                    # Evaluation section
                    modules.basic_functions.subsection_evaluate('cp', modules.correlation.evaluate.functions.keys(),
                                                                False),

                ],
                    className='c_l3',
                ),

                # Plot
                html.Div([
                    dcc.Graph(id='cp_plot', config=fig_config),
                ],
                    className='c_plot',
                ),
            ]),

            # %% Parametric plot
            dcc.Tab(label='Parametric plot', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div([
                    # Select time series
                    modules.basic_functions.subsection_select_two_variables('pp', 'Plot parametric'),

                    # add to plot section
                    modules.basic_functions.subsection_add_to_plot('pp', modules.parametric.add.functions.keys(),
                                                                   False),

                    # Evaluation section
                    modules.basic_functions.subsection_evaluate('pp', modules.parametric.evaluate.functions.keys(),
                                                                False),

                ],
                    className='c_l3',
                ),

                # Plot
                html.Div([
                    dcc.Graph(id='pp_plot', config=fig_config),
                ],
                    className='c_plot',
                ),
            ],
                    className='tab'
                    ),
        ],
            className='tabs'
        ),
    ],
        className='border margin',
        id='c_tabs'
    )
])


# Callbacks
# %% load data
# dataset selection -> loop dropdown options
@app.callback(Output('ld_loop', 'options'),
              [Input('ld_dataset', 'value')])
def update_ld_log(value):
    # wait until the dataset is selected
    if value is None:
        raise PreventUpdate

    global hdf_data, info

    # close before open new
    if hdf_data is not None:
        hdf_data.close()

    hdf_data = pd.HDFStore(data_folder / value, mode='r')
    info = hdf_data['info']

    # get loop names
    loops = info.index
    if 'default' in loops:
        loops = loops.drop('default')

    # create dictionary with the loop options
    return [{'label': i, 'value': i} for i in loops]


# loop selection -> print loop information
@app.callback(Output('ld_log', 'children'),
              [Input('ld_loop', 'value')],
              [State('ld_dataset', 'value')])
def update_ld_log(ld_lp, ld_ds):
    # wait until loop is selected
    if ld_lp is None:
        raise PreventUpdate

    global info_loop, selected_loop
    selected_loop = ld_lp
    info_loop = info.loc[info.index == ld_lp]

    to_log(message=f'You are working with dataset ** {ld_ds.split(".")[0]} ** and loop ** {ld_lp} **')
    st = [f'Loop info:\n\n']

    # get loop information
    for l in loop_info_print:
        info_l = info_loop[l].values[0]
        # if it is a number less than zero, check the map
        if (not isinstance(info_l, str)) and (info_l < 0):
            info_l = map_codes[str(int(info_l))]

        if l == 'year of origin':
            try:
                info_l = int(info_l)
            except:
                pass

        # add info to the markdown string
        st.append('**' + l.capitalize() + ':** ' + (str(info_l)).capitalize() + '  \n')

    # from string to markdown
    return [dcc.Markdown(''.join(st))]


# loop selection -> measurements dropdown options
@app.callback(
    [Output('dp_plot_proc_var', 'options'),
     Output('td_plot_add_var', 'options'),
     Output('td_ev_var', 'options'),
     Output('fd_plot_add_var', 'options'),
     Output('fd_ev_var', 'options'),
     Output('pp_sel1', 'options'),
     Output('pp_sel2', 'options'),
     Output('cp_sel1', 'options'),
     Output('cp_sel2', 'options')],
    [Input('ld_loop', 'value')])
def update_measurements_dropdown(loop):
    # wait until loop is selected
    if loop is None:
        raise PreventUpdate

    global info
    info_loop_local = info.loc[info.index == loop]

    # get the available measurements
    global measurements
    measurements = info_loop_local['measurements'].values[0]

    # create dropdown options with the available measurements for all the required dropdowns
    return [[{'label': m, 'value': m} for m in measurements]] * 9


# %% data processing
# data processing apply function -> data processing apply parameters
@app.callback([Output('dp_plot_proc_par', 'children'),
               Output('dp_plot_proc_info', 'title')],
              [Input('dp_plot_proc', 'value')])
def update_dp_apply(value):
    return modules.basic_functions.parameters_update(value, modules.data_processing.functions, 'dp', '')


# loop selection -> plot data processing
@app.callback(
    [Output('dp_plot', 'figure'),
     Output('dp_ts', 'placeholder'),
     Output('dp_plot_rg_start', 'value'),
     Output('dp_plot_rg_end', 'value')],
    [Input('dp_bt_orig', 'n_clicks'),
     Input('dp_bt_rg', 'n_clicks'),
     Input('dp_bt_ap', 'n_clicks')],
    [State('ld_loop', 'value'),
     State('dp_ts', 'value'),
     State('dp_plot_rg_start', 'value'),
     State('dp_plot_rg_end', 'value'),
     State('dp_plot_proc', 'value'),
     State('dp_plot_proc_var', 'value'),
     State('dp_plot_proc_par', 'children'),
     State('dp_plot', 'figure'),
     State('dp_plot', 'relayoutData')])
def update_plot_dp(click_plot, click_rg, click_ap, loop, ts, start, end, func, var, par_ch, fig, fig_layout):
    # wait until loop is selected
    if loop is None:
        raise PreventUpdate

    global dp_click_plot, dp_click_rg, dp_click_ap
    global Ts, measurements, info_loop, info_general, x_start, x_end, hdf_data
    global data_processed, data_interp, data_range

    # if click on "plot original data"
    if click_plot != dp_click_plot:
        to_log(default='plotting', par='sampled data')
        info_general = hdf_data['general_info']

        # get data
        data = {}

        # if each variable has it own timestamp
        if info_general['standard_type'].values[0] == 'separated_files_standard':
            for m in measurements:
                data[m] = (hdf_data.get(loop + '/' + m))

        else:
            data_l = hdf_data.get(loop)
            for m in measurements:
                data[m] = data_l[[m]].rename(columns={m: 'Values'})

        # if time variable is timestamp, change to time starting in zero
        if info_general['time_stamp'].values[0]:
            data = modules.basic_functions.time_stamp_to_array(data, measurements)

        # change Ts
        data_interp, Ts = modules.basic_functions.change_ts(info_loop, ts, measurements, data)

        # plot
        fig = modules.basic_functions.fig_template(rows=2, xaxes_title='time (s)',
                                                   yaxes_title=['original', 'processed'])

        for m in measurements:
            # original data
            trace = go.Scatter(x=data[m].index, y=data[m]['Values'], name=m)
            fig.append_trace(trace, 1, 1)

            # interpolated data
            trace_dp = go.Scatter(x=data_interp[m].index, y=data_interp[m]['Values'], name=m + '_proc')
            fig.append_trace(trace_dp, 2, 1)

        x_start, x_end = 2 * [None]
        data_range = copy.deepcopy(data_interp)

        # update buttons status
        dp_click_plot = click_plot
        dp_click_rg = click_rg
        dp_click_ap = click_ap

    # if click on "set range"
    if click_rg != dp_click_rg:
        to_log(default='plotting', par='with new range')

        # get start and end points
        x0, x1 = modules.basic_functions.get_start_and_end(start, end, fig_layout, data_interp, measurements)

        # set range for the processed data
        for ind_p, p in enumerate(fig['data']):
            if p['xaxis'] == 'x2':
                m = fig['data'][ind_p]['name'][:2]
                data_range[m] = data_interp[m].loc[x0:x1]
                fig['data'][ind_p]['y'] = data_range[m]['Values'].values
                fig['data'][ind_p]['x'] = data_range[m].index

        x_start, x_end = round(x0, 2), round(x1, 2)
        dp_click_rg = click_rg

    # if click on "apply preprocessing"
    if click_ap != dp_click_ap:
        if func is None:
            to_log(default='sel_function')
        elif var is None:
            to_log(default='sel_variable')

        else:
            to_log(default='plotting', par='processed data')

            for ind, i in enumerate(fig['data']):
                if i['name'] == var + '_proc':
                    ind_true = ind

            x = data_range[var].index
            y = data_range[var]['Values']

            # get parameters
            par = modules.basic_functions.get_parameters(par_ch)

            # run function
            func_exe = getattr(modules.data_processing, func)
            x_apply, y_apply = func_exe(x, y, par)

            # replace old data by new
            fig['data'][ind_true]['y'] = y_apply
            fig['data'][ind_true]['x'] = x_apply

            dp_click_ap = click_ap

    # save processed data internally
    data_processed = {}
    for d_ind, d in enumerate(fig['data']):
        if d['xaxis'] == 'x2':
            data_processed[d['name'][:2]] = pd.DataFrame(d['y'], d['x'], ['Values'])

    return fig, 'Default Ts = ' + str(Ts), x_start, x_end


# processed data -> save
@app.callback(Output('dp_save_nooutput', 'hidden'),
              [Input('dp_save_bt', 'n_clicks')],
              [State('dp_save_dataset', 'value'),
               State('dp_save_name', 'value'),
               State('dp_save_description', 'value'),
               State('dp_save_short_description', 'value')])
def save_data(click, dataset, name, desc, short_desc):
    if (click is None) or (data_processed is None):
        raise PreventUpdate

    # prepare data
    y = []
    for k in data_processed.keys():
        y.append(data_processed[k]['Values'].values)

    y = np.asarray(y).transpose()
    x = data_processed[k].index
    data_save = pd.DataFrame(y, x, measurements)

    # load dataset
    store = pd.HDFStore(data_folder / dataset)
    info_save = store['info']

    # if name not given, create default, which is the original loop name + '_index'
    if (name is None) or (name == ''):
        name_l = selected_loop + '_1'
        ind = 2
        while name_l in info_save.index:
            name_l = selected_loop + '_' + str(ind)
            ind += 1
        name = name_l

    if desc is None:
        desc = ''

    if short_desc is None:
        short_desc = ''

    # store data
    store.put(name, data_save)

    # if the dataset has a save with the same name, overwrite
    if name in info_save.index:
        info_save = info_save.drop(name)

    # get loop info from the original data
    info_l = info_loop

    info_l.index = [name]
    info_l['description'] = [desc]
    info_l['short description'] = [short_desc]
    info_l['ts'] = Ts

    # include loop info to info in dataset
    info_l = info_l.append(info_save, sort=True)

    # store info
    store.put('info', info_l)
    store.close()

    to_log(message='Processed data saved')

    return True


# %% time domain
# time domain add function -> time domain add parameters
@app.callback([Output('td_plot_add_par', 'children'),
               Output('td_plot_add_func_info', 'title')],
              [Input('td_plot_add_func', 'value')])
def update_td_add_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.time_domain.add.functions, 'td', 'plot_add')
    return ch, res


# processed data -> plot time domain + add to plot
@app.callback(
    Output('td_plot', 'figure'),
    [Input('td_plot_add_bt', 'n_clicks'),
     Input('td_plot_bt', 'n_clicks')],
    [State('td_plot_add_func', 'value'),
     State('td_plot_add_var', 'value'),
     State('td_plot_add_par', 'children'),
     State('td_plot_add_name', 'value'),
     State('td_plot', 'figure')])
def update_plot_td(click_add, click_plot, func, var, par_ch, name, fig):
    # wait until loop is selected
    if click_plot is None:
        raise PreventUpdate

    global td_click_plot
    global td_click_add

    if td_click_plot != click_plot:
        to_log(default='plotting', par='time domain')

        fig = modules.basic_functions.fig_template(rows=2, xaxes_title='time (s)',
                                                   yaxes_title=['OP and MV', 'PV and SP'])

        # get data and plot
        for k in data_processed.keys():
            trace = go.Scatter(x=data_processed[k].index, y=data_processed[k]['Values'], name=k)
            pt = 1 if (k == 'MV') or (k == 'OP') else 2
            fig.append_trace(trace, pt, 1)

        # update button status
        td_click_plot = click_plot
        td_click_add = click_add

    # Add more traces
    if td_click_add != click_add:
        if func is None:
            to_log(default='sel_function')
        elif var is None:
            to_log(default='sel_variable')

        else:
            to_log(default='adding_trace', par='time domain')

            # get data
            y = data_processed[var]['Values'].values
            x = data_processed[var].index

            # get parameters
            x_add, y_add, name = modules.basic_functions.subsection_add_to_plot_run(func, var, par_ch, name,
                                                                                    modules.time_domain.add, x, y)

            # selected subplot according to the variable
            pt = 1 if (var == 'MV') or (var == 'OP') else 2

            # add to plot
            dc = dict(x=x_add, y=y_add, name=name, type='scatter', xaxis='x' + str(pt), yaxis='y' + str(pt))
            fig['data'].append(dc)

    return fig


# time domain evaluate function -> time domain evaluate parameters
@app.callback([Output('td_ev_par', 'children'),
               Output('td_ev_func_info', 'title')],
              [Input('td_ev_func', 'value')])
def update_td_ev_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.time_domain.evaluate.functions, 'td', 'ev')
    return ch, res


# processed data -> time domain evaluation
@app.callback(Output('td_ev_res', 'children'),
              [Input('td_ev_bt', 'n_clicks')],
              [State('td_ev_func', 'value'),
               State('td_ev_var', 'value'),
               State('td_ev_par', 'children'),
               State('td_ev_res', 'children')])
def update_td_res(click, func, var, par_ch, res):
    # wait until the button is clicked
    if click is None:
        raise PreventUpdate

    elif func is None:
        to_log(default='sel_function')
    elif var is None:
        to_log(default='sel_variable')

    else:
        to_log(default='running_ev', par='time domain')
        # get data
        y = data_processed[var]['Values'].values
        x = data_processed[var].index

        res = modules.basic_functions.subsection_evaluate_print(func, var, par_ch, res, modules.time_domain.evaluate,
                                                                x, y)
        return res


# %% frequency domain
# frequency domain add function -> frequency domain add parameters
@app.callback([Output('fd_plot_add_par', 'children'),
               Output('fd_plot_add_func_info', 'title')],
              [Input('fd_plot_add_func', 'value')])
def update_fd_add_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.frequency_domain.add.functions, 'fd', 'plot_add')
    return ch, res


# processed data -> plot frequency domain + add to plot
@app.callback(
    Output('fd_plot', 'figure'),
    [Input('fd_plot_add_bt', 'n_clicks'),
     Input('fd_plot_bt', 'n_clicks')],
    [State('fd_plot_add_func', 'value'),
     State('fd_plot_add_var', 'value'),
     State('fd_plot_add_par', 'children'),
     State('fd_plot_add_name', 'value'),
     State('fd_plot', 'figure')])
def update_plot_fd(click_add, click_plot, func, var, par_ch, name, fig):
    # wait until loop is selected
    if click_plot is None:
        raise PreventUpdate

    global fd_click_plot
    global fd_click_add

    if fd_click_plot != click_plot:
        to_log(default='plotting', par='frequency domain')

        fig = modules.basic_functions.fig_template(rows=1, xaxes_title='length(data)/Ts Hz', yaxes_title=['amplitude'])

        # plot frequency domain of the processed data
        for k in data_processed.keys():
            # get data
            y = data_processed[k]['Values'].values
            y = y - np.mean(y)

            # frequency domain
            y = abs(np.fft.fft(y))
            x = np.arange(len(y))

            # plot spectrum
            fig.add_trace(go.Scatter(x=x, y=y, name=k))

        # update figure to first half
        fig.update_xaxes(range=[0, int(max(x) / 2)])

        # update button values
        fd_click_plot = click_plot
        fd_click_add = click_add

    # add more traces
    if fd_click_add != click_add:
        if func is None:
            to_log(default='sel_function')
        elif var is None:
            to_log(default='sel_variable')

        else:
            to_log(default='adding_trace', par='frequency domain')

            # get data
            for ind, i in enumerate(fig['data']):
                if i['name'] == var:
                    ind_true = ind

            y = fig['data'][ind_true]['y']
            x = fig['data'][ind_true]['x']

            x_add, y_add, name = modules.basic_functions.subsection_add_to_plot_run(func, var, par_ch, name,
                                                                                    modules.frequency_domain.add, x, y)

            # add plot
            dc = dict(x=x_add, y=y_add, name=name, type='scatter')
            fig['data'].append(dc)

    return fig


# frequency domain evaluate function -> frequency domain evaluate parameters
@app.callback([Output('fd_ev_par', 'children'),
               Output('fd_ev_func_info', 'title')],
              [Input('fd_ev_func', 'value')])
def update_fd_ev_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.frequency_domain.evaluate.functions, 'fd', 'ev')
    return ch, res


# processed data -> frequency domain evaluation
@app.callback(Output('fd_ev_res', 'children'),
              [Input('fd_ev_bt', 'n_clicks')],
              [State('fd_ev_func', 'value'),
               State('fd_ev_var', 'value'),
               State('fd_ev_par', 'children'),
               State('fd_ev_res', 'children'),
               State('fd_plot', 'figure')])
def update_fd_res(click, func, var, par_ch, res, fig):
    # wait until the button is clicked
    if click is None:
        raise PreventUpdate

    elif func is None:
        to_log(default='sel_function')
    elif var is None:
        to_log(default='sel_variable')

    else:
        to_log(default='running_ev', par='frequency domain')

        # get data
        for ind, i in enumerate(fig['data']):
            if i['name'] == var:
                ind_true = ind

        y = fig['data'][ind_true]['y']
        x = fig['data'][ind_true]['x']

        res = modules.basic_functions.subsection_evaluate_print(func, var, par_ch, res,
                                                                modules.frequency_domain.evaluate, x, y)
        return res


# %% correlation plot
# correlation plot add function -> correlation  add parameters
@app.callback([Output('cp_plot_add_par', 'children'),
               Output('cp_plot_add_func_info', 'title')],
              [Input('cp_plot_add_func', 'value')])
def update_fd_add_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.correlation.add.functions, 'cp', 'plot_add')
    return ch, res


# processed data -> correlation plot
@app.callback(
    Output('cp_plot', 'figure'),
    [Input('cp_plot_add_bt', 'n_clicks'),
     Input('cp_plot_bt', 'n_clicks')],
    [State('cp_sel1', 'value'),
     State('cp_sel2', 'value'),
     State('cp_plot_add_func', 'value'),
     State('cp_plot_add_par', 'children'),
     State('cp_plot_add_name', 'value'),
     State('cp_plot', 'figure')])
def update_plot_cp(click_add, click_plot, sel1, sel2, func, par_ch, name, fig):
    # wait until loop is selected
    if click_plot is None:
        raise PreventUpdate

    global cp_click_plot
    global cp_click_add

    if cp_click_plot != click_plot:
        if (sel1 is None) or (sel1 == '') or (sel2 is None) or (sel2 == ''):
            to_log(default='sel_variables')

        else:
            to_log(default='plotting', par='correlation plot')

            # get data
            x = data_processed[sel1]['Values'].values
            y = data_processed[sel2]['Values'].values

            # remove mean
            x = x - np.mean(x)
            y = y - np.mean(y)

            # calculate the correlation
            cor = np.correlate(x, y, 'full')
            cor_norm = cor / np.max(cor)
            t = np.arange(len(cor)) - int(len(cor) / 2)

            # plot
            fig = modules.basic_functions.fig_template(rows=1, xaxes_title='lag', yaxes_title=['amplitude'])
            fig.add_trace(go.Scatter(x=t, y=cor_norm, name='cor_plot'))

            # update button values
            cp_click_plot = click_plot
            cp_click_add = click_add

    # add more traces
    if cp_click_add != click_add:
        if func is None:
            to_log(default='sel_function')

        else:
            to_log(default='adding_trace', par='time domain')

            # get data
            for ind, i in enumerate(fig['data']):
                if i['name'] == 'cor_plot':
                    ind_true = ind

            y = fig['data'][ind_true]['y']
            x = fig['data'][ind_true]['x']

            x_add, y_add, name = modules.basic_functions.subsection_add_to_plot_run(func, '', par_ch, name,
                                                                                    modules.correlation.add, x,
                                                                                    y)

            # add plot
            dc = dict(x=x_add, y=y_add, name=name, type='scatter')
            fig['data'].append(dc)

    return fig


# correlation evaluate function -> correlation evaluate parameters
@app.callback([Output('cp_ev_par', 'children'),
               Output('cp_ev_func_info', 'title')],
              [Input('cp_ev_func', 'value')])
def update_fd_ev_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.correlation.evaluate.functions, 'cp', 'ev')
    return ch, res


# processed data -> correlation evaluation
@app.callback(Output('cp_ev_res', 'children'),
              [Input('cp_ev_bt', 'n_clicks')],
              [State('cp_ev_func', 'value'),
               State('cp_ev_par', 'children'),
               State('cp_ev_res', 'children'),
               State('cp_plot', 'figure')])
def update_cp_res(click, func, par_ch, res, fig):
    # wait until the button is clicked
    if click is None:
        raise PreventUpdate

    elif func is None:
        to_log(default='sel_function')

    else:
        to_log(default='running_ev', par='correlation plot')

        # get data
        y = fig['data'][0]['y']
        x = fig['data'][0]['x']

        res = modules.basic_functions.subsection_evaluate_print(func, '', par_ch, res, modules.correlation.evaluate, x,
                                                                y)
        return res


# %% parametric plot
# processed data -> parametric plot
@app.callback(
    Output('pp_plot', 'figure'),
    [Input('pp_plot_add_bt', 'n_clicks'),
     Input('pp_plot_bt', 'n_clicks')],
    [State('pp_sel1', 'value'),
     State('pp_sel2', 'value'),
     State('pp_plot_add_func', 'value'),
     State('pp_plot_add_par', 'children'),
     State('pp_plot_add_name', 'value'),
     State('pp_plot', 'figure')])
def update_plot_pp(click_add, click_plot, sel1, sel2, func, par_ch, name, fig):
    # wait until loop is selected
    if click_plot is None:
        raise PreventUpdate

    global pp_click_plot
    global pp_click_add

    if pp_click_plot != click_plot:
        if (sel1 is None) or (sel1 == '') or (sel2 is None) or (sel2 == ''):
            to_log(default='sel_variables')

        else:
            to_log(default='plotting', par='parametric plot')

            # get the data
            x = data_processed[sel1]['Values'].values
            y = data_processed[sel2]['Values'].values

            # plot
            fig = modules.basic_functions.fig_template(rows=1, xaxes_title=sel1, yaxes_title=[sel2])
            fig.add_trace(go.Scatter(x=x, y=y, name='cor_plot'))

            # update button values
            pp_click_plot = click_plot
            pp_click_add = click_add

    # add more traces
    if pp_click_add != click_add:
        if func is None:
            to_log(default='sel_function')

        else:
            to_log(default='adding_trace', par='time domain')

            # get data
            for ind, i in enumerate(fig['data']):
                if i['name'] == 'cor_plot':
                    ind_true = ind

            y = fig['data'][ind_true]['y']
            x = fig['data'][ind_true]['x']

            x_add, y_add, name = modules.basic_functions.subsection_add_to_plot_run(func, '', par_ch, name,
                                                                                    modules.parametric.add, x,
                                                                                    y)

            # add plot
            dc = dict(x=x_add, y=y_add, name=name, type='scatter')
            fig['data'].append(dc)

    return fig


# parametric plot evaluate function -> parametric plot evaluate parameters
@app.callback([Output('pp_ev_par', 'children'),
               Output('pp_ev_func_info', 'title')],
              [Input('pp_ev_func', 'value')])
def update_fd_ev_par(value):
    ch, res = modules.basic_functions.parameters_update(value, modules.parametric.evaluate.functions, 'pp', 'ev')
    return ch, res


# processed data -> parametric plot evaluation
@app.callback(Output('pp_ev_res', 'children'),
              [Input('pp_ev_bt', 'n_clicks')],
              [State('pp_ev_func', 'value'),
               State('pp_ev_par', 'children'),
               State('pp_ev_res', 'children'),
               State('pp_plot', 'figure')])
def update_fd_res(click, func, par_ch, res, fig):
    # wait until the button is clicked
    if click is None:
        raise PreventUpdate

    elif func is None:
        to_log(default='sel_function')

    else:
        to_log(default='running_ev', par='parametric plot')

        # get data
        y = fig['data'][0]['y']
        x = fig['data'][0]['x']

        res = modules.basic_functions.subsection_evaluate_print(func, '', par_ch, res, modules.parametric.evaluate, x,
                                                                y)
        return res


# %% general log
@app.callback(
    Output('gl_log', 'children'),
    [Input('lg_interval', 'n_intervals')])
def update_global_log(n):
    global general_log, general_log_collect

    # if there is message to be printed
    if general_log_collect:
        for m in general_log_collect:
            general_log.append(m)
        general_log_collect = []
    return [dcc.Markdown('\n\n'.join(general_log))]


if __name__ == '__main__':
    app.run_server(debug=True)
