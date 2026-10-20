from task import *
import json
from astar_with_waypoints import *

with open("problem/q.json") as f:
    questions = json.load(f)

if __name__ == '__main__':
    print("cooking……")
    for q in questions:
        task=Task(q)
        result=task.operate()
        run(q,result)