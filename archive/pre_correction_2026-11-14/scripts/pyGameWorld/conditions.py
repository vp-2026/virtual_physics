import numpy as np
from .constants import *
from .object import *

__all__ = ["PGCond_AnyInGoal", "PGCond_SpecificInGoal", "PGCond_AnyTouch",
           "PGCond_SpecificTouch", "PGCond_ManyInGoal"]


def _point_in_poly(x, y, poly):
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-12)
            if x_int > x:
                inside = not inside
    return inside


def _dist_point_to_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    cx, cy = ax + t * vx, ay + t * vy
    return np.hypot(px - cx, py - cy)


def _min_dist_to_polygon_edges(px, py, poly):
    return min(
        _dist_point_to_segment(px, py, ax, ay, bx, by)
        for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1])
    )


def _ball_fully_in_polygon(obj, goal_poly, eps=1e-6):
    pos = obj.getPos()
    cx, cy = float(pos[0]), float(pos[1])
    radius = float(obj.radius)
    if not _point_in_poly(cx, cy, goal_poly):
        return False
    return _min_dist_to_polygon_edges(cx, cy, goal_poly) + eps >= radius


def _object_fully_in_goal(obj, goal):
    goal_poly = [(float(v[0]), float(v[1])) for v in goal.getVertices()]

    if getattr(obj, "type", None) == "Ball":
        return _ball_fully_in_polygon(obj, goal_poly)

    if hasattr(obj, "getVertices"):
        try:
            obj_poly = [(float(v[0]), float(v[1])) for v in obj.getVertices()]
            return all(_point_in_poly(x, y, goal_poly) for x, y in obj_poly)
        except Exception:
            pass

    pos = obj.getPos()
    return _point_in_poly(float(pos[0]), float(pos[1]), goal_poly)

class PGCond_Base(object):

    def __init__(self):
        self.goal = self.obj = self.parent = self.dur = None

    def _getTimeIn(self):
        return -1

    def remainingTime(self):
        ti = self._getTimeIn()
        if ti == -1:
            return None
        curtime = self.parent.time - ti
        return max(self.dur - curtime, 0)

    def isWon(self):
        return self.remainingTime() == 0

    def attachHooks(self):
        raise NotImplementedError("Cannot attach hooks from base condition object")


class PGCond_AnyInGoal(PGCond_Base):

    def __init__(self, goalname, duration, parent, exclusions = []):
        self.type = "AnyInGoal"
        self.won = False
        self.goal = goalname
        self.excl = exclusions
        self.dur = duration
        self.ins = {}
        self.hasTime = True
        self.parent = parent

    def _goesIn(self, obj, goal):
        if (goal.name == self.goal and \
                    (not obj.name in self.ins.keys()) and \
                    (not obj.name in self.excl)):
            self.ins[obj.name] = self.parent.time

    def _goesOut(self, obj, goal):
        if (goal.name == self.goal and \
            obj.name in self.ins.keys() and \
                    (not goal.pointIn(obj.position))):
            del self.ins[obj.name]

    def attachHooks(self):
        self.parent.setGoalCollisionBegin(self._goesIn)
        self.parent.setGoalCollisionEnd(self._goesOut)

    def _getTimeIn(self):
        if len(self.ins) == 0:
            return -1
        mintime = min(min(self.ins.values()), self.parent.time)
        return mintime

class PGCond_ManyInGoal(PGCond_Base):

    def __init__(self, goalname, objlist, duration, parent):
        self.type = "ManyInGoal"
        self.won = False
        self.goal = goalname
        self.objlist = objlist
        self.objsin = []
        self.dur = duration
        self.tin = -1
        self.hasTime = True
        self.parent = parent

    def _goesIn(self, obj, goal):
        if (goal.name == self.goal and
            obj.name in self.objlist and
            obj.name not in self.objsin):
            self.objsin.append(obj.name)
            if len(self.objsin) == 1:
                self.tin = self.parent.time

    def _goesOut(self, obj, goal):
        if (goal.name == self.goal and
            obj.name in self.objsin):
            self.objsin.remove(obj.name)
            if len(self.objsin) == 0:
                self.tin = -1

    def attachHooks(self):
        self.parent.setGoalCollisionBegin(self._goesIn)
        self.parent.setGoalCollisionEnd(self._goesOut)

    def _getTimeIn(self):
        return self.tin


class PGCond_SpecificInGoal(PGCond_Base):

    def __init__(self, goalname, objname, duration, parent):
        self.type = "SpecificInGoal"
        self.won = False
        self.goal = goalname
        self.obj = objname
        self.dur = duration
        self.tin = -1
        self.hasTime = True
        self.parent = parent

    def _goesIn(self, obj, goal):
        if goal.name == self.goal and obj.name == self.obj:
            self.tin = self.parent.time

    def _goesOut(self, obj, goal):
        if goal.name == self.goal and obj.name == self.obj and (not goal.pointIn(obj.position)):
            self.tin = -1

    def attachHooks(self):
        self.parent.setGoalCollisionBegin(self._goesIn)
        self.parent.setGoalCollisionEnd(self._goesOut)

    def _getTimeIn(self):
        try:
            goal = self.parent.getObject(self.goal)
            obj = self.parent.getObject(self.obj)
        except Exception:
            self.tin = -1
            return self.tin

        if _object_fully_in_goal(obj, goal):
            if self.tin == -1:
                self.tin = self.parent.time
        else:
            self.tin = -1
        return self.tin


class PGCond_AnyTouch(PGCond_Base):

    def __init__(self, objname, duration, parent):
        self.type = "AnyTouch"
        self.won = False
        self.goal = objname
        self.dur = duration
        self.tin = -1
        self.hasTime = True
        self.parent = parent

    def _beginTouch(self, obj, goal):
        if obj.name == self.goal or goal.name == self.goal:
            self.tin = self.parent.time

    def _endTouch(self, obj, goal):
        if obj.name == self.goal or goal.name == self.goal:
            sefl.tin = -1

    def attachHooks(self):
        self.parent.setSolidCollisionBegin(self._beginTouch)
        self.parent.setSolidCollisionEnd(self._endTouch)

    def _getTimeIn(self):
        return self.tin

class PGCond_SpecificTouch(PGCond_Base):

    def __init__(self, objname1, objname2, duration, parent):
        self.type = "SpecificTouch"
        self.won = False
        self.o1 = objname1
        self.o2 = objname2
        self.dur = duration
        self.tin = -1
        self.hasTime = True
        self.parent = parent

    def _beginTouch(self, obj1, obj2):
        if (obj1.name == self.o1 and obj2.name == self.o2) or \
            (obj1.name == self.o2 and obj2.name == self.o1):
            self.tin = self.parent.time

    def _endTouch(self, obj1, obj2):
        if (obj1.name == self.o1 and obj2.name == self.o2) or \
            (obj1.name == self.o2 and obj2.name == self.o1):
            self.tin = -1

    def attachHooks(self):
        self.parent.setSolidCollisionBegin(self._beginTouch)
        self.parent.setSolidCollisionEnd(self._endTouch)

    def _getTimeIn(self):
        return self.tin
