from flask import Flask, send_from_directory
import os
import subprocess
import sys
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

last_update_time = None
UPDATE_INTERVAL = timedelta(minutes=1)

@app.route('/')
def dashboard():
    global last_update_time
    current_time = datetime.now(pytz.UTC)
    
    should_update = (
        last_update_time is None or 
        (current_time - last_update_time) > UPDATE_INTERVAL
    )
    
    if should_update:
        script_path = os.path.join(BASE_DIR, 'Tasks_Analysis.py')
        try:
            subprocess.run([sys.executable, script_path], check=True)
            last_update_time = current_time
            app.logger.info(f"Dashboard updated at {last_update_time}")
        except subprocess.CalledProcessError as e:
            app.logger.error(f"Error running analysis script: {str(e)}")
            return "Error generating dashboard. Please check the logs.", 500

    return send_from_directory(os.path.join(BASE_DIR, 'templates'), 'dashboard.html')

@app.route('/<task_html>')
def task_page(task_html):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), task_html)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
