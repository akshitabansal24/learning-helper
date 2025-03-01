from google.appengine.api import mail
import os

def send_mail():
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    print(project_id)
    address = "akshitabansal24@gmail.com"

    try:
        mail.send_mail(
            sender=f"demo-app@{project_id}.appspotmail.com",
            to=address,
            subject="App Engine Outgoing Email",
            body="rftghyjuskdfjs"
        )
    except Exception as e:
        print(f"Sending mail to {address} failed with exception {e}.")
        return f"Exception {e} when sending mail to {address}.", 500

    print(f"Successfully sent mail to {address}.")
    # return f"Successfully sent mail to {address}.", 201


send_mail()