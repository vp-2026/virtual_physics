import os

modulepath = os.path.join(os.path.dirname(__file__), 'node_modules')

# --- (base_context is unchanged except for escaping its own braces) ---
base_context = r'''
module.paths.push('{0}')
var pg = require('PhysicsGaming')
var npg = require('NoisyPG')
function zeroIfUndef(x) {{
    return (typeof(x) === 'undefined' || x === null) ? 0 : x
}}
function isntEmpty(obj) {{
  return Object.keys(obj).length > 0
}}
function applyNoise(world, noisedict) {{
    var nps = zeroIfUndef(noisedict.noise_position_static)
    var npm = zeroIfUndef(noisedict.noise_position_moving)
    var ncd = zeroIfUndef(noisedict.noise_collision_direction)
    var nce = zeroIfUndef(noisedict.noise_collision_elasticity)
    var ng = zeroIfUndef(noisedict.noise_gravity)
    var nof = zeroIfUndef(noisedict.noise_object_friction)
    var nod = zeroIfUndef(noisedict.noise_object_density)
    var noe = zeroIfUndef(noisedict.noise_object_elasticity)
    return npg.noisifyWorld(world, nps, npm, ncd, nce, ng, nof, nod, noe)
}}
function runGW(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    while (running) {{
        w.step(stepSize)
        t += stepSize
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    if (returnNewWorld) {{
        return [w.checkEnd(), t, returnWorld]
    }} else {{
        return [w.checkEnd(), t]
    }}
}}
function stepGW(worldDict, stepSize) {{
    var w = pg.loadFromDict(worldDict)
    w.step(stepSize)
    return w.toDict()
}}
function getGWPath(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    var pathdict = {{}}
    var tracknames = []
    for (onm in w.objects) {{
        var o = w.objects[onm]
        if (!o.isStatic()) {{
            tracknames.push(onm)
            pathdict[onm] = [o.getPos()]
        }}
    }}
    while (running) {{
        w.step(stepSize)
        t += stepSize
        for (var i = 0; i < tracknames.length; i++) {{
            onm = tracknames[i]
            pathdict[onm].push(w.objects[onm].getPos())
        }}
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    if (returnNewWorld) {{
        return [pathdict, w.checkEnd(), t, returnWorld]
    }} else {{
        return [pathdict, w.checkEnd(), t]
    }}
}}
function getGWPathAndRot(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    var pathdict = {{}}
    var tracknames = []
    for (onm in w.objects) {{
        var o = w.objects[onm]
        if (!o.isStatic()) {{
            tracknames.push(onm)
            pathdict[onm] = [[o.getPos()], [o.getRot()]]
        }}
    }}
    while (running) {{
        w.step(stepSize)
        t += stepSize
        for (var i = 0; i < tracknames.length; i++) {{
            onm = tracknames[i]
            pathdict[onm][0].push(w.objects[onm].getPos())
            pathdict[onm][1].push(w.objects[onm].getRot())
        }}
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    if (returnNewWorld) {{
        return [pathdict, w.checkEnd(), t, returnWorld]
    }} else {{
        return [pathdict, w.checkEnd(), t]
    }}
}}
function getGWStatePath(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    var pathdict = {{}}
    var tracknames = []
    for (onm in w.objects) {{
        var o = w.objects[onm]
        if (!o.isStatic()) {{
            tracknames.push(onm)
            pathdict[onm] = [[o.getPos()[0], o.getPos()[1], o.getRot(), o.getVel()[0], o.getVel()[1]]]
        }}
    }}
    while (running) {{
        w.step(stepSize)
        t += stepSize
        for (var i = 0; i < tracknames.length; i++) {{
            onm = tracknames[i]
            pathdict[onm].push([w.objects[onm].getPos()[0], w.objects[onm].getPos()[1], w.objects[onm].getRot(), w.objects[onm].getVel()[0], w.objects[onm].getVel()[1]])
        }}
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    if (returnNewWorld) {{
        return [pathdict, w.checkEnd(), t, returnWorld]
    }} else {{
        return [pathdict, w.checkEnd(), t]
    }}
}}
function getGWGeomPath(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    var pathdict = {{}}
    var tracknames = []
    function toGeom(o) {{
        if (o.type === "Poly") {{
            return(o.getVertices())
        }} else if (o.type === "Ball") {{
            return([o.getPos(), o.getRadius()])
        }} else if (o.type === "Container" || o.type === "Compound") {{
            return (o.getPolys())
        }} else {{
            console.log("Shape type not found: ", o.type)
            return
        }}
    }}
    for (onm in w.objects) {{
        var o = w.objects[onm]
        if (!o.isStatic()) {{
            tracknames.push(onm)
            pathdict[onm] = [[o.type, toGeom(o), o.getVel()]]
        }}
    }}
    while (running) {{
        w.step(stepSize)
        t += stepSize
        for (var i = 0; i < tracknames.length; i++) {{
            onm = tracknames[i]
            o = w.objects[onm]
            pathdict[onm].push([o.type, toGeom(o), o.getVel()])
        }}
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    if (returnNewWorld) {{
        return [pathdict, w.checkEnd(), t, returnWorld]
    }} else {{
        return [pathdict, w.checkEnd(), t]
    }}
}}
function getGWCollisionPath(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    var pathdict = {{}}
    var tracknames = []
    for (onm in w.objects) {{
        var o = w.objects[onm]
        if (!o.isStatic()) {{
            tracknames.push(onm)
            pathdict[onm] = [o.getPos()]
        }}
    }}
    while (running) {{
        w.step(stepSize)
        t += stepSize;
        for (var i = 0; i < tracknames.length; i++) {{
            onm = tracknames[i];
            pathdict[onm].push(w.objects[onm].getPos())
        }}
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    collisions = w.getCollisionEvents()
    if (returnNewWorld) {{
        return [pathdict, collisions, w.checkEnd(), t, returnWorld]
    }} else {{
        return [pathdict, collisions, w.checkEnd(), t]
    }}
}}
function getGWCollisionPathAndRot(worldDict, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = pg.loadFromDict(worldDict)
    if (typeof(noiseDict) !== 'undefined' && isntEmpty(noiseDict)) {{
        w = applyNoise(w, noiseDict)
    }}
    if (returnNewWorld) {{
        var returnWorld = w.toDict()
    }}
    var running = true
    var t = 0
    var pathdict = {{}}
    var tracknames = []
    for (onm in w.objects) {{
        var o = w.objects[onm]
        if (!o.isStatic()) {{
            tracknames.push(onm)
            pathdict[onm] = [[o.getPos()], [o.getRot()]]
        }}
    }}
    while (running) {{
        w.step(stepSize)
        t += stepSize
        for (var i = 0; i < tracknames.length; i++) {{
            onm = tracknames[i]
            pathdict[onm][0].push(w.objects[onm].getPos())
            pathdict[onm][1].push(w.objects[onm].getRot())
        }}
        if (w.checkEnd() || (t >= maxtime)) {{
            running = false
        }}
    }}
    collisions = w.getCollisionEvents()
    if (returnNewWorld) {{
        return [pathdict, collisions, w.checkEnd(), t, returnWorld]
    }} else {{
        return [pathdict, collisions, w.checkEnd(), t]
    }}
}}
'''

# ──────────────────────────────────────────────────────────────────────────────
# Now append the small collision‐checker snippet, escaping its braces as well:
collision_context = base_context + r'''
var world = pg.loadFromDict(JSON.parse('{1}'))
world.step(.000001)
function checkSinglePlaceCollide(verts, position) {{
    return world.checkCollision(position, verts)
}}
function checkCircleCollide(position, radius) {{
    return world.checkCircleCollision(position, radius)
}}
function checkMultiPlaceCollide(polys, position) {{
    for (var i=0; i < polys.length; i++) {{
        if (checkSinglePlaceCollide(polys[i], position)) return true
    }}
    return false
}}
'''

# ──────────────────────────────────────────────────────────────────────────────
# Finally, append the “addTool(...)” & “addBall(...)” code, with all braces doubled:
context = collision_context + r'''
function addTool(worldDict, toolDef, pos) {{
    if (typeof(toolDef) === 'undefined') return worldDict;

    var rawPolys = toolDef.polys;
    var dens     = toolDef.density;
    var fric     = toolDef.friction;
    var elast    = toolDef.elasticity;

    var placedPolys = [];
    for (var i = 0; i < rawPolys.length; i++) {{
        var singlePoly = rawPolys[i];
        var movedPoly   = [];
        for (var j = 0; j < singlePoly.length; j++) {{
            var vx = singlePoly[j][0] + pos[0];
            var vy = singlePoly[j][1] + pos[1];
            movedPoly.push([vx, vy]);
        }}
        placedPolys.push(movedPoly);
    }}

    worldDict.objects["PLACED"] = {{
        type:       "Compound",
        color:      "blue",
        density:    dens,
        friction:   fric,
        elasticity: elast,
        polys:      placedPolys
    }};

    return worldDict;
}}

function addBall(worldDict, pos, rad) {{
    if (typeof(pos) === 'undefined') return worldDict;
    worldDict.objects["PLACED"] = {{
        type:     "Ball",
        color:    "blue",
        density:  1,
        position: pos,
        radius:   rad
    }};
    return worldDict;
}}

function runGWPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return runGW(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}

function getGWPathPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return getGWPath(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}

function getGWPathAndRotPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return getGWPathAndRot(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}

function getGWStatePathPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return getGWStatePath(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}

function getGWCollisionPathPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return getGWCollisionPath(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}

function getGWCollisionPathAndRotPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return getGWCollisionPathAndRot(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}

function getGWGeomPathPlacement(worldDict, toolDef, pos, maxtime, stepSize, noiseDict, returnNewWorld) {{
    var w = addTool(worldDict, toolDef, pos);
    return getGWGeomPath(w, maxtime, stepSize, noiseDict, returnNewWorld);
}}
'''
