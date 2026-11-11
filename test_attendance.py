from datetime import datetime

marked = set()          # names already marked today

def mark_attendance(name):
    if name == "Unknown" or name in marked:
        return
    marked.add(name)
    now = datetime.now()
    with open("attendance.csv", "a") as f:
        f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
    print("Marked:", name)

mark_attendance("homie")
mark_attendance("homie")
mark_attendance("Unknown")