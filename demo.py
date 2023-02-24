from Insurance.pipeline.batch_prediction import start_batch_prediction

file_path = r"C:\Users\shara\OneDrive\Documents\Sharath\ineuron\002_Insurance Predictor\Insurance-Predictor\insurance.csv"

if __name__ == '__main__':
    try:
        output = start_batch_prediction(input_file_path=file_path)
    except Exception as e:
        print(e)