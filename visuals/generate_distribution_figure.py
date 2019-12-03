'''
confusion matrix generator wit plotly express
@todo: not looks elegant create colorscale
'''
import numpy as np 
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from optparse import OptionParser

def generate_data_svg_plots(data_mode,metadata_file):
    data = pd.read_csv(metadata_file)
    data =data.sort_values(by=['race'])
    
    fig = px.histogram(data, x="race", 
                    histfunc="sum", 
                    color=data.race,
                    color_discrete_sequence=['#ED553B','#F6D55C','#3CAEA3','#20639B'], 
                    barmode="group")

    fig.update_yaxes(title_text="Number of Images")
    fig.update_xaxes(title_text="Race")
    fig.update_layout(title_text="RFW %s Image's Race Distribution"%data_mode)
    file_name = "../source/RFW_{}_Images_Race_Distribution.svg".format(data_mode)
    fig.write_image(file_name)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--mf', '--metadata_file', dest='metadata_file', default='../data/RFW_Train40k_Images_Metada.csv',
                      help='predicted label file')
    parser.add_option('--dm', '--data_mode', dest='data_mode', default='Train',
                      help='data mode would be Train,Test,Validation')

    (options, args) = parser.parse_args()
    generate_data_svg_plots(options.data_mode,options.metadata_file)
