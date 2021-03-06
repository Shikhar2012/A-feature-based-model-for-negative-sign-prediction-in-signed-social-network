function EVAL = Evaluate(ACTUAL,PREDICTED)

idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx))
tn = sum(ACTUAL(~idx)==PREDICTED(~idx))
fp = n-tn
fn = p-tp
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL = ["Accuracy =" 1- accuracy   "Precision=" 1-precision "Recall=" 1-recall "f_measure=" 1-f_measure];