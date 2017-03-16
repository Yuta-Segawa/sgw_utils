import argparse
import glob, os
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_scores(cm, verbosity=True):
    """Calculate precision, recall, and F-measure from confusion matrix. 

    :param cm: Confusion matrix of ground truth and predictions. 
    :param verbosity: Verbosity of showing status. 
    :return: Tuple of precision, recall, and F-measure as numpy arrays. 
    """
    true_positives = np.float32(np.diag(cm))
    false_positives = np.float32(np.sum(cm, axis=0) - true_positives)
    false_negatives = np.float32(np.sum(cm, axis=1) - true_positives)

    Precision = []
    Recall = []
    F_measure = []
    for label in range(0, cm.shape[0]):
        prec = true_positives[label] / (true_positives[label] + false_positives[label])
        rec = true_positives[label] / (true_positives[label] + false_negatives[label])
        f_m = 2 * prec * rec / (prec + rec)
        Precision.append(prec)
        Recall.append(rec)
        F_measure.append(f_m)
        if verbosity:
            print("Class %d:" % label)
            f = '{0}: {1:.3f}'
            print f.format("Precision", prec)
            print f.format("Recall", rec)
            print f.format("F-measure", f_m)
    return np.array(Precision), np.array(Recall), np.array(F_measure)

def display_scores(dirpath, verbosity=False):
    """Display summarized scores. 

    :param dirpath: Directory containing score files named as '*_cm.npy', '*_prec.npy', '*_rec.npy', and , '*_f_m.npy'. 
    :param verbosity: Verbosity of showing attributes of scores. 
    """

    precision_label = ""
    recall_label = ""
    f_measure_label = ""
    accuracy_label = ""

    if verbosity:
        precision_label = "Precision\t"
        recall_label = "Recall\t"
        f_measure_label = "F_measure\t"
        accuracy_label = "Accuracy\t"

    cms = np.array([np.load(fn) for fn in sorted(glob.glob(os.path.join(dirpath, "*_cm.npy")))])
    precs = np.array([np.load(fn) for fn in sorted(glob.glob(os.path.join(dirpath, "*_prec.npy")))])
    recs = np.array([np.load(fn) for fn in sorted(glob.glob(os.path.join(dirpath, "*_rec.npy")))])
    f_ms = np.array([np.load(fn) for fn in sorted(glob.glob(os.path.join(dirpath, "*_f_m.npy")))])

    # exception numbers (NaN) are assigned to zero
    precs[np.isnan(precs)] = 0.0
    recs[np.isnan(recs)] = 0.0
    f_ms[np.isnan(f_ms)] = 0.0

    print "dipath: ", dirpath
    print "total cm: "
    print cms.sum(axis=0)

    if verbosity:
        print "Averaged scores for classes: "
    macro_precs = precs.mean(axis=0) * 100.0
    macro_recs = recs.mean(axis=0) * 100.0
    macro_f_ms = f_ms.mean(axis=0) * 100.0
    macro_precs_str = precision_label + "\t".join(["%.1f"%val for val in macro_precs])
    macro_recs_str = recall_label + "\t".join(["%.1f"%val for val in macro_recs])
    macro_f_ms_str = f_measure_label + "\t".join(["%.1f"%val for val in macro_f_ms])

    print macro_precs_str
    print macro_recs_str
    print macro_f_ms_str

    if verbosity:
        print "Scores averaged over classes: "
    classwise_macro_prec = macro_precs.mean(axis=0)
    classwise_macro_rec = macro_recs.mean(axis=0)
    classwise_macro_f_m = macro_f_ms.mean(axis=0)
    accuracy = np.mean([ float(np.diag(cm).sum()) / float(cm.sum()) for cm in cms]) * 100.0

    print precision_label + "%.1f" % classwise_macro_prec
    print recall_label + "%.1f" % classwise_macro_rec
    print f_measure_label + "%.1f" % classwise_macro_f_m
    print accuracy_label + "%.1f" % accuracy

def score_saver(output_dir, y_pred, y_true, identifier="evaldata", skipcase=True, verbosity=True):
    """Save scores of precision, recall, F-measure calculated by prediction results and ground truth. 

    :param output_dir: Directory for outputting scores: 

        - <output_dir>/<identifier>_cm.npy: confusion matrix
        - <output_dir>/<identifier>_prec.npy: precision
        - <output_dir>/<identifier>_rec.npy: recall
        - <output_dir>/<identifier>_f_m.npy: F-measure

    :param y_pred: Result of prediction. 
    :param y_true: Ground truth corresponding to the y_pred. 
    :param identifier: Identifier of output file name. 
    :param skipcase: Flag to skip the calculation and saving when the output scores are already existing. 
    :pamar verbosity: Verbosity of showing status(except skipcase info). 
    """

    cm_fn = os.path.join(output_dir, "%s_cm.npy" % identifier)
    prec_fn = os.path.join(output_dir, "%s_prec.npy" % identifier)
    rec_fn = os.path.join(output_dir, "%s_rec.npy" % identifier)
    f_m_fn = os.path.join(output_dir, "%s_f_m.npy" % identifier)

    # check whether the scores have been already calculated
    if os.path.exists(f_m_fn) and skipcase:
        print "[I]The scores have been already calculated. Skip on '%s'. " % identifier

    else:
        cm = confusion_matrix(y_true, y_pred)
        if verbosity:
            print cm
        p, r, f = calculate_scores(cm, verbosity=verbosity)

        np.save(cm_fn, np.array(cm))
        np.save(prec_fn, np.array(p))
        np.save(rec_fn, np.array(r))
        np.save(f_m_fn, np.array(f))


if __name__ == '__main__':
    def main():
        try:
            label = np.load(args.label_path)
            classified = np.load(args.classified_path)
        except:
            quit()

        cm = confusion_matrix(label, classified)

        print cm
        _, _, f_m = calculate_scores(cm)
        output_f_m_filename = args.label_path.rstrip("_labele.npy")+"_F-measure.npy"
        np.save(output_f_m_filename, f_m)
    def ArgParse():
        parser = argparse.ArgumentParser(description='Evaluate on the Keras platform.')

        # file or folder path: 
        parser.add_argument('-l', '--label_path', dest='label_path', type=str, 
                            default="model/fine-tuned_vgg-16.model",
                            help='[string]A model file including a graph and weights.')
        parser.add_argument('-c', '--classified_path', dest='classified_path', type=str, 
                            default="model/fine-tuned_vgg-16.model",
                            help='[string]A model file including a graph and weights.')


        return parser.parse_args()

        args = ArgParse()

    main()



