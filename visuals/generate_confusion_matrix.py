'''
confusion matrix generator wit plotly express
@todo: not looks elegant create colorscale
'''
import numpy as np 
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from optparse import OptionParser

def draw_confusion_matrix(prediction_file,label_file,output_file):
    y_pred = np.load(prediction_file)
    y_true = np.load(label_file)

    z = confusion_matrix(y_true, y_pred)
    z_normalized = z.astype('int') / z.sum(axis=1)[:, np.newaxis]*100
    z_normalized = np.around(z_normalized, decimals=2)

    x=["African", "Asian", "Caucasian", "  Indian"]
    y=["African", "Asian", "Caucasian", "  Indian"]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_normalized, colorscale='YlOrRd')
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(
        font=dict(
            family="'Open Sans'",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.write_image(output_file)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--pf', '--prediction_file', dest='prediction_file', default='source/wsdan_predictions.npy',
                      help='predicted label file')
    parser.add_option('--lf', '--label_file', dest='label_file', default='source/wsdan_ground_truth.npy',
                      help='ground truth label file .npy format')

    parser.add_option('--of', '--output_file', dest='output_file', default='wsdan_confusion.pdf',
                      help='output confusion matrix file')

    (options, args) = parser.parse_args()
    draw_confusion_matrix(options.prediction_file,options.label_file,options.output_file)
