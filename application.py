from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')

    else:
        data=CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX=request.form.get('SEX'),
            EDUCATION=request.form.get('EDUCATION'),
            MARRIAGE=request.form.get('MARRIAGE'),
            AGE=float(request.form.get('AGE')),
            PAY_1=float(request.form.get('PAY_1')),
            PAY_2=float(request.form.get('PAY_2')),
            PAY_3=float(request.form.get('PAY_3')),
            PAY_4=float(request.form.get('PAY_4')),
            PAY_5=float(request.form.get('PAY_5')),
            PAY_6=float(request.form.get('PAY_6')),
            BILL_AMT1=float(request.form.get('BILL_AMT1')),
            BILL_AMT2=float(request.form.get('BILL_AMT2')),
            BILL_AMT3=float(request.form.get('BILL_AMT3')),
            BILL_AMT4=float(request.form.get('BILL_AMT4')),
            BILL_AMT5=float(request.form.get('BILL_AMT5')),
            BILL_AMT6=float(request.form.get('BILL_AMT6')),
            PAY_AMT1=float(request.form.get('PAY_AMT1')),
            PAY_AMT2=float(request.form.get('PAY_AMT2')),
            PAY_AMT3=float(request.form.get('PAY_AMT3')),
            PAY_AMT4=float(request.form.get('PAY_AMT4')),
            PAY_AMT5=float(request.form.get('PAY_AMT5')),
            PAY_AMT6=float(request.form.get('PAY_AMT6'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        if pred[0] ==1:
            prediction_results ="The credit card holder will be Defaulter in the next month"
        
        else:
            prediction_results = "The Credit card holder will not be Defaulter in the next month"

        return render_template('result.html',prediction_result=prediction_results)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)