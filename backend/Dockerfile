FROM python:3.10.6-slim

RUN apt-get update && apt-get install -y cron

WORKDIR /app

COPY requirements.txt .
COPY setup.py .

RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    pip install .

COPY senti_ai/ senti_ai/

RUN echo '#!/bin/bash\n\
python -c "from senti_ai.ml_logic.data import combine_all_data_and_save; combine_all_data_and_save()"' > /app/run_daily_task.sh

RUN chmod +x /app/run_daily_task.sh

RUN echo "0 \${SCHEDULE_HOUR:-0} * * * /app/run_daily_task.sh >> /var/log/cron.log 2>&1" > /etc/cron.d/data_updater
RUN chmod 0644 /etc/cron.d/data_updater

RUN touch /var/log/cron.log

CMD crontab /etc/cron.d/data_updater && cron -f
