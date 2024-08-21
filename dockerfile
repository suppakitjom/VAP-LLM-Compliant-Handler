FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.lambda_handler" ]
