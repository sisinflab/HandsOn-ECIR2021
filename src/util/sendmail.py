import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

me = "sisinflabtome@gmail.com"
my_password = "Sisinf2020lab"
you = "merrafelice@gmail.com"

msg = MIMEMultipart('alternative')
msg['From'] = me
msg['To'] = you


def sendmail(mail_object, message):
    msg['Subject'] = mail_object
    html = '<html><body><p>{0}</p></body></html>'.format(message)
    part2 = MIMEText(html, 'html')

    msg.attach(part2)
    s = smtplib.SMTP_SSL('smtp.gmail.com')
    s.login(me, my_password)

    s.sendmail(me, you, msg.as_string())
    s.quit()