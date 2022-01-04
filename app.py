import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# for colors
import matplotlib.colors as mcolors

from drawing_tools import *
from giraffe_tools import *

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Ray casting visualization"),
    dcc.Graph(id='graph', style={'width': '90wh', 'height': '60vh'}),
    
    html.Div([
        html.Div([
            
            # changes to setting and ray casted 
            html.Label([ "u (theta)",
                dcc.Slider(
                    id='u-slider', 
                    min=0, max=1,
                    value=0.00,
                    marks={str(val) : str(val) for val in [0.00, 0.25, 0.50, 0.75]},
                    step=0.01, tooltip = { 'always_visible': True }
                ), ]),
            html.Label([ "v (phi)",
                dcc.Slider(
                    id='v-slider', 
                    min=0, max=1,
                    value=0.25,
                    marks={str(val) : str(val) for val in [0.00, 0.25, 0.50, 0.75]},
                    step=0.01, tooltip = { 'always_visible': True }
                ), ]),
            html.Label([ "r (sphere radius)",
                dcc.Slider(
                    id='r-slider', 
                    min=0, max=5,
                    value=2.713,
                    marks={str(val) : str(val) for val in [0.000, 1.000, 2.713, 5.000]},
                    step=0.001, tooltip = { 'always_visible': True }
                ), ]),
            html.Label([ "depth range",
                dcc.RangeSlider(
                    id='depth-range-slider', 
                    min=0, max=10,
                    value=[0.5, 6],
                    marks={str(val) : str(val) for val in [0.5 * i for i in range(21)]},
                    step=0.1, tooltip = { 'always_visible': True }
                ), ])
        ], style = {'width' : '48%', 'display' : 'inline-block'}),
        
        html.Div([
            # changes to visual appearance
            
            # axis scale
            html.Div([
                html.Label([ "world axis size",
                    html.Div([
                        dcc.Input(id='world-axis-size-input',
                                  value=1.5,
                                  type='number', style={'width': '50%'}
                                 )
                    ]),
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                html.Label([ "camera axis size",
                    html.Div([
                        dcc.Input(id='camera-axis-size-input',
                                  value=0.3,
                                  type='number', style={'width': '50%'}
                                 )
                    ]),
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
            ]),
            
            # color
            html.Div([
                html.Label([ "camera color",
                html.Div([
                    dcc.Dropdown(id='camera-color-input',
                                 clearable=False,
                              value='yellow',
                              options=[
                                     {'label': c, 'value': c}
                                     for (c, _) in mcolors.CSS4_COLORS.items()
                                 ], style={'width': '80%'}
                             )
                ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "ray color",
                    html.Div([
                        dcc.Dropdown(id='ray-color-input',
                                     clearable=False,
                                  value='yellow',
                                  options=[
                                         {'label': c, 'value': c}
                                         for (c, _) in mcolors.CSS4_COLORS.items()
                                     ], style={'width': '80%'}
                                 )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "frustrum color",
                    html.Div([
                        dcc.Dropdown(id='frustrum-color-input',
                                     clearable=False,
                                  value='orange',
                                  options=[
                                         {'label': c, 'value': c}
                                         for (c, _) in mcolors.CSS4_COLORS.items()
                                     ], style={'width': '80%'}
                                 )
                    ])
                ],  style = {'width' : '32%', 'float' : 'left', 'display' : 'inline-block'}),
            ]),
            
            # colorscale
            html.Div([
                html.Label([ "sphere colorscale",
                    html.Div([
                        dcc.Dropdown(id='sphere-colorscale-input',
                                     clearable=False,
                                     value='greys',
                                     options=[
                                         {'label': c, 'value': c}
                                         for c in px.colors.named_colorscales()
                                     ], style={'width': '80%'}
                                    )
                    ])
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "xy-plane colorscale",
                    html.Div([
                        dcc.Dropdown(id='xy-plane-colorscale-input',
                                     clearable=False,
                                     value='greys',
                                     options=[
                                         {'label': c, 'value': c}
                                         for c in px.colors.named_colorscales()
                                     ], style={'width': '80%'}
                                    )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
            ]),
            
            # opacity 
            html.Div([
                html.Label([ "sphere opacity",
                    html.Div([
                        dcc.Input(id='sphere-opacity-input',
                                  value=0.2,
                                  type='number', style={'width': '50%'}
                                 )
                    ])
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                            
                html.Label([ "xy-plane opacity",            
                    html.Div([
                        dcc.Input(id='xy-plane-opacity-input',
                                  value=0.8,
                                  type='number', style={'width': '50%'}
                                 )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "frustrum opacity",
                    html.Div([
                        dcc.Input(id='frustrum-opacity-input',
                                  value=0.3,
                                  type='number', style={'width': '50%'}
                                 )
                    ])
                ],  style = {'width' : '32%', 'float' : 'left', 'display' : 'inline-block'}),
            ]),
            
        ], style = {'width' : '48%', 'float' : 'right', 'display' : 'inline-block'}),
            
    ]),
    
])

@app.callback(
    Output('graph', 'figure'),
    Input("u-slider", "value"),
    Input("v-slider", "value"),
    
    Input("r-slider", "value"),
    Input("depth-range-slider", "value"),
    
    Input("world-axis-size-input", "value"),
    Input("camera-axis-size-input", "value"),
    
    Input("camera-color-input", "value"),
    Input("ray-color-input", "value"),
    Input("frustrum-color-input", "value"),
    
    Input('sphere-colorscale-input', "value"),
    Input('xy-plane-colorscale-input', "value"),
    
    Input("sphere-opacity-input", "value"),
    Input("xy-plane-opacity-input", "value"),
    Input("frustrum-opacity-input", "value"),
)

def update_figure(u, v, 
                  r, depth_range,
                  world_axis_size, camera_axis_size,
                  camera_color, ray_color, frustrum_color,
                  sphere_colorscale, xy_plane_colorscale,
                  sphere_opacity, xy_plane_opacity, frustrum_opacity                  
                 ):
    # sphere
    fig = draw_sphere(r=r, sphere_colorscale=sphere_colorscale, sphere_opacity=sphere_opacity)

    # change figure size
#     fig.update_layout(autosize=False, width = 500, height=500)

    # draw axes in proportion to the proportion of their ranges
    fig.update_layout(scene_aspectmode='data')

    # xy plane
    fig = draw_XYplane(fig, xy_plane_colorscale, xy_plane_opacity,
                       x_range=[-depth_range[1], depth_range[1]], y_range=[-depth_range[1], depth_range[1]])

    # show world coordinate system (X, Y, Z positive direction)
    fig = draw_XYZworld(fig, world_axis_size=world_axis_size)

    pixels_world, camera_world, world_mat, p_i = giraffe(u=u, v=v, r=r, depth_range=depth_range)

    #  draw camera at init (with its cooridnate system)
    fig = draw_cam_init(fig, world_mat, 
                        camera_axis_size=camera_axis_size, camera_color=camera_color)

    # draw all rays
    fig = draw_all_rays(fig, p_i, ray_color=ray_color)

    # draw near&far frustrum with rays connecting the corners
    fig = draw_ray_frus(fig, p_i, frustrum_color=frustrum_color, frustrum_opacity=frustrum_opacity,
                        at=[0, 8, -1])
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)