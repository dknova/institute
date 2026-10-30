/* ------------------------------------------------------------------ */
/*  Setup                                                              */
/* ------------------------------------------------------------------ */
var canvas = document.getElementById('stage');
var gl = canvas.getContext('webgl', { antialias: true, alpha: false });

if (!gl) {
    document.body.insertAdjacentHTML(
        'afterbegin',
        '<p style="position:fixed; z-index:9; padding:1rem">WebGL is not supported in this browser.</p>'
    );
    throw new Error('WebGL initialization failed.');
}

var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
var PILL_Y = -18.5;   // capsule sits at the "dosing" section

/* ------------------------------------------------------------------ */
/*  Math helpers (column-major 4x4 matrices)                           */
/* ------------------------------------------------------------------ */
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }

function smoothstep(a, b, x) {
    var t = clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

function identity() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); }

function translate(x, y, z) { var m = identity(); m[12] = x; m[13] = y; m[14] = z; return m; }
function scale(x, y, z)     { var m = identity(); m[0] = x; m[5] = y; m[10] = z; return m; }

function rotateX(a) {
    var m = identity(), c = Math.cos(a), s = Math.sin(a);
    m[5] = c; m[9] = -s; m[6] = s; m[10] = c;
    return m;
}
function rotateZ(a) {
    var m = identity(), c = Math.cos(a), s = Math.sin(a);
    m[0] = c; m[4] = -s; m[1] = s; m[5] = c;
    return m;
}

function mul() {
    var r = arguments[0];
    for (var n = 1; n < arguments.length; n++) {
        var a = r, b = arguments[n], o = new Float32Array(16);
        for (var c = 0; c < 4; c++) {
            for (var row = 0; row < 4; row++) {
                var s = 0;
                for (var k = 0; k < 4; k++) s += a[k * 4 + row] * b[c * 4 + k];
                o[c * 4 + row] = s;
            }
        }
        r = o;
    }
    return r;
}

function perspective(fovDeg, aspect, near, far) {
    var f = 1 / Math.tan(fovDeg * Math.PI / 360), m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f;
    m[10] = (far + near) / (near - far); m[11] = -1;
    m[14] = 2 * far * near / (near - far);
    return m;
}

function normalize3(v) {
    var l = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) || 1;
    v[0] /= l; v[1] /= l; v[2] /= l;
    return v;
}

/* ------------------------------------------------------------------ */
/*  Shaders                                                            */
/* ------------------------------------------------------------------ */
function compile(type, source) {
    var s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
    return s;
}

function makeProgram(vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.bindAttribLocation(p, 1, 'aNormal');
    gl.bindAttribLocation(p, 2, 'aColor');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p));
    return p;
}

function locations(p, names) {
    var o = {};
    names.forEach(function (n) { o[n] = gl.getUniformLocation(p, n); });
    return o;
}

var FRAG_PRECISION = [
    '#ifdef GL_FRAGMENT_PRECISION_HIGH',
    'precision highp float;',
    '#else',
    'precision mediump float;',
    '#endif'
].join('\n');

/* Capsule + blood cells: uniform colour, uParent * uModel */
var litProg = makeProgram(
    [
        'precision highp float;',
        'attribute vec3 aPos;',
        'attribute vec3 aNormal;',
        'uniform mat4 uProj, uView, uParent, uModel;',
        'varying vec3 vN;',
        'varying vec3 vW;',
        'void main(){',
        '  mat4 m = uParent * uModel;',
        '  vec4 world = m * vec4(aPos, 1.0);',
        '  vW = world.xyz;',
        '  vN = (m * vec4(aNormal, 0.0)).xyz;',
        '  gl_Position = uProj * uView * world;',
        '}'
    ].join('\n'),
    [
        FRAG_PRECISION,
        'varying vec3 vN;',
        'varying vec3 vW;',
        'uniform vec3 uColor, uCam, uKeyPos, uRimPos;',
        'void main(){',
        '  vec3 n  = normalize(vN);',
        '  vec3 v  = normalize(uCam - vW);',
        '  vec3 lk = normalize(uKeyPos - vW);',
        '  vec3 lr = normalize(uRimPos - vW);',
        '  float dk = max(dot(n, lk), 0.0);',
        '  float dr = max(dot(n, lr), 0.0);',
        '  float spec = pow(max(dot(n, normalize(lk + v)), 0.0), 48.0);',
        '  vec3 col = uColor * (0.3 + 0.9 * dk)',
        '           + vec3(0.35, 0.55, 0.9) * uColor * dr * 0.35',
        '           + vec3(1.0, 0.9, 0.75) * spec * 0.5;',
        '  float d = length(uCam - vW);',
        '  float fog = exp(-d * d * 0.0012);',
        '  gl_FragColor = vec4(mix(vec3(0.051, 0.102, 0.165), col, fog), 1.0);',
        '}'
    ].join('\n')
);
var L = locations(litProg, ['uProj', 'uView', 'uParent', 'uModel', 'uColor', 'uCam', 'uKeyPos', 'uRimPos']);

/* DNA ribbons + rungs: per-vertex colour, two-sided silky lighting */
var dnaProg = makeProgram(
    [
        'precision highp float;',
        'attribute vec3 aPos;',
        'attribute vec3 aNormal;',
        'attribute vec3 aColor;',
        'uniform mat4 uProj, uView;',
        'varying vec3 vN;',
        'varying vec3 vW;',
        'varying vec3 vC;',
        'void main(){',
        '  vW = aPos; vN = aNormal; vC = aColor;',
        '  gl_Position = uProj * uView * vec4(aPos, 1.0);',
        '}'
    ].join('\n'),
    [
        FRAG_PRECISION,
        'varying vec3 vN;',
        'varying vec3 vW;',
        'varying vec3 vC;',
        'uniform vec3 uCam, uKeyPos, uRimPos;',
        'void main(){',
        '  vec3 v = normalize(uCam - vW);',
        '  vec3 n = normalize(vN);',
        '  if (dot(n, v) < 0.0) n = -n;',
        '  vec3 lk = normalize(uKeyPos - vW);',
        '  vec3 lr = normalize(uRimPos - vW);',
        '  float dk = max(dot(n, lk), 0.0);',
        '  float dr = max(dot(n, lr), 0.0);',
        '  float spec = pow(max(dot(n, normalize(lk + v)), 0.0), 40.0);',
        '  float fres = pow(1.0 - max(dot(n, v), 0.0), 2.5);',
        '  vec3 col = vC * (0.35 + 0.85 * dk)',
        '           + vec3(0.40, 0.50, 0.95) * vC * dr * 0.45',
        '           + vec3(1.0, 0.92, 0.96) * spec * 0.55',
        '           + vec3(0.55, 0.60, 1.0) * fres * 0.25;',
        '  float d = length(uCam - vW);',
        '  float fog = exp(-d * d * 0.0012);',
        '  gl_FragColor = vec4(mix(vec3(0.051, 0.102, 0.165), col, fog), 1.0);',
        '}'
    ].join('\n')
);
var D = locations(dnaProg, ['uProj', 'uView', 'uCam', 'uKeyPos', 'uRimPos']);

var flatProg = makeProgram(
    [
        'precision highp float;',
        'attribute vec3 aPos;',
        'uniform mat4 uProj, uView, uModel;',
        'uniform float uSize;',
        'void main(){',
        '  gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);',
        '  gl_PointSize = uSize;',
        '}'
    ].join('\n'),
    [
        'precision mediump float;',
        'uniform vec4 uColor;',
        'void main(){ gl_FragColor = uColor; }'
    ].join('\n')
);
var F = locations(flatProg, ['uProj', 'uView', 'uModel', 'uColor', 'uSize']);

/* ------------------------------------------------------------------ */
/*  Static meshes for the capsule and blood cells                      */
/* ------------------------------------------------------------------ */
function makeMesh(vertices, indices) {
    var vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    var ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    return { vbo: vbo, ibo: ibo, count: indices.length };
}

function buildSphere(lat, lon) {
    var v = [], idx = [], i, j;
    for (i = 0; i <= lat; i++) {
        var theta = i * Math.PI / lat;
        for (j = 0; j <= lon; j++) {
            var phi = j * 2 * Math.PI / lon;
            var x = Math.sin(theta) * Math.cos(phi), y = Math.cos(theta), z = Math.sin(theta) * Math.sin(phi);
            v.push(x, y, z, x, y, z);
        }
    }
    for (i = 0; i < lat; i++) {
        for (j = 0; j < lon; j++) {
            var a = i * (lon + 1) + j, b = a + lon + 1;
            idx.push(a, b, a + 1, b, b + 1, a + 1);
        }
    }
    return makeMesh(v, idx);
}

function buildCylinder(seg) {
    var v = [], idx = [], j;
    for (j = 0; j <= seg; j++) {
        var phi = j * 2 * Math.PI / seg, c = Math.cos(phi), s = Math.sin(phi);
        v.push(c,  0.5, s, c, 0, s);
        v.push(c, -0.5, s, c, 0, s);
    }
    for (j = 0; j < seg; j++) {
        var a = j * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    return makeMesh(v, idx);
}

var sphere   = buildSphere(24, 32);
var cylinder = buildCylinder(24);

/* ------------------------------------------------------------------ */
/*  Flowing DNA: two ribbon strands + thin rungs along a wavy spine    */
/* ------------------------------------------------------------------ */
var DNA = {
    Y0: 7, LEN: 20,         // spine runs from y=7 down to y=-13
    PITCH: 3.2,             // length of one full turn
    STEPS: 240,             // samples along each ribbon
    RUNGS: 125,             // base-pair rungs
    R0: 1.25,               // helix radius
    HW: 0.34,               // ribbon half-width (along the axis)
    RW: 0.014,              // rung half-thickness
    OFF: 2.7                // angle between strands (unequal = major/minor groove)
};

var RIB_VERTS   = 2 * DNA.STEPS * 2;
var TOTAL_VERTS = RIB_VERTS + DNA.RUNGS * 8;
var dnaVerts    = new Float32Array(TOTAL_VERTS * 9);   // pos(3) normal(3) colour(3)

var dnaBuf = gl.createBuffer();
var dnaIbo = gl.createBuffer();
var dnaCount = 0;
(function () {
    var idx = [], k, i, j;
    for (k = 0; k < 2; k++) {
        var vb = k * DNA.STEPS * 2;
        for (i = 0; i < DNA.STEPS - 1; i++) {
            var a = vb + i * 2;
            idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
        }
    }
    for (j = 0; j < DNA.RUNGS; j++) {
        var rb = RIB_VERTS + j * 8;
        for (var q = 0; q < 2; q++) {
            var v = rb + q * 4;
            idx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
        }
    }
    dnaCount = idx.length;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, dnaIbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW);
})();

var dnaCX = 0, dnaAmp = 1;       // set every frame
var _p0 = [0,0,0], _p1 = [0,0,0];
var FR = { c: [0,0,0], T: [0,0,0], N: [0,0,0], B: [0,0,0], phi: 0, R: 0, hw: 0, tap: 1 };

/* The centre line: a fixed, gentle S-curve (no time-based movement). */
function spine(u, t, p, o) {
    var y = DNA.Y0 - DNA.LEN * u;
    o[0] = dnaCX + dnaAmp * (Math.sin(y * 0.30) + 0.35 * Math.sin(y * 0.75));
    o[1] = y;
    o[2] = -3 + 4.5 * u + 0.8 * Math.sin(y * 0.22 + t * 0.4 + 1.0);
}

/* Position, tangent and a twist-free frame (N, B) at spine parameter u */
function frameAt(u, t, p, rs) {
    spine(u, t, p, FR.c);
    spine(u - 0.002, t, p, _p0);
    spine(u + 0.002, t, p, _p1);

    var T = FR.T;
    T[0] = _p1[0] - _p0[0]; T[1] = _p1[1] - _p0[1]; T[2] = _p1[2] - _p0[2];
    normalize3(T);

    var N = FR.N;                          // cross((0,0,1), T)
    N[0] = -T[1]; N[1] = T[0]; N[2] = 0;
    normalize3(N);

    var B = FR.B;                          // cross(T, N)
    B[0] = T[1] * N[2] - T[2] * N[1];
    B[1] = T[2] * N[0] - T[0] * N[2];
    B[2] = T[0] * N[1] - T[1] * N[0];

    var y = FR.c[1];
    FR.phi = (2 * Math.PI / DNA.PITCH) * (DNA.LEN * u) + p * 8 + 0.6 * Math.sin(y * 0.35 - t * 0.9);
    FR.tap = smoothstep(0, 0.05, u) * (1 - smoothstep(0.88, 1, u));   // ends melt into a point
    FR.R   = DNA.R0 * rs * FR.tap * (1 + 0.1 * Math.sin(y * 0.45 + t * 0.8));
    FR.hw  = DNA.HW * rs * FR.tap * (0.9 + 0.2 * Math.sin(y * 0.6 - t * 0.7));
}

/* Indigo at the top, violet in the middle, crimson-magenta at the bottom */
var C_TOP = [0.26, 0.30, 0.80], C_MID = [0.55, 0.22, 0.70], C_BOT = [0.93, 0.24, 0.46];
function gradient(u, out) {
    var a = smoothstep(0.0, 0.5, u), b = smoothstep(0.4, 1.0, u);
    for (var i = 0; i < 3; i++) {
        var m = C_TOP[i] + (C_MID[i] - C_TOP[i]) * a;
        out[i] = m + (C_BOT[i] - m) * b;
    }
    return out;
}

var vo = 0;
function put(x, y, z, nx, ny, nz, c) {
    var v = dnaVerts;
    v[vo++] = x;  v[vo++] = y;  v[vo++] = z;
    v[vo++] = nx; v[vo++] = ny; v[vo++] = nz;
    v[vo++] = c[0]; v[vo++] = c[1]; v[vo++] = c[2];
}

var _col = [0,0,0], _col2 = [0,0,0];

function buildDNA(t, p, rs) {
    vo = 0;
    var k, i, j;

    /* the two ribbon strands */
    for (k = 0; k < 2; k++) {
        for (i = 0; i < DNA.STEPS; i++) {
            var u = i / (DNA.STEPS - 1);
            frameAt(u, t, p, rs);
            var ang = FR.phi + (k ? DNA.OFF : 0);
            var ca = Math.cos(ang), sa = Math.sin(ang);
            var N = FR.N, B = FR.B, T = FR.T, c = FR.c;
            var rx = ca * N[0] + sa * B[0], ry = ca * N[1] + sa * B[1], rz = ca * N[2] + sa * B[2];
            var px = c[0] + FR.R * rx, py = c[1] + FR.R * ry, pz = c[2] + FR.R * rz;

            gradient(u, _col);
            if (k) { _col[0] = _col[0] * 0.9 + 0.06; _col[1] = _col[1] * 0.9 + 0.06; _col[2] = _col[2] * 0.95 + 0.04; }

            put(px - T[0] * FR.hw, py - T[1] * FR.hw, pz - T[2] * FR.hw, rx, ry, rz, _col);
            put(px + T[0] * FR.hw, py + T[1] * FR.hw, pz + T[2] * FR.hw, rx, ry, rz, _col);
        }
    }

    /* thin rungs joining the strands (two crossed quads so they read from any angle) */
    for (j = 0; j < DNA.RUNGS; j++) {
        var uj = (j + 0.5) / DNA.RUNGS;
        frameAt(uj, t, p, rs);
        var N2 = FR.N, B2 = FR.B, T2 = FR.T, c2 = FR.c, R = FR.R * 0.98;
        var a0 = FR.phi, a1 = FR.phi + DNA.OFF;
        var c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);

        var x0 = c2[0] + R * (c0 * N2[0] + s0 * B2[0]), y0 = c2[1] + R * (c0 * N2[1] + s0 * B2[1]), z0 = c2[2] + R * (c0 * N2[2] + s0 * B2[2]);
        var x1 = c2[0] + R * (c1 * N2[0] + s1 * B2[0]), y1 = c2[1] + R * (c1 * N2[1] + s1 * B2[1]), z1 = c2[2] + R * (c1 * N2[2] + s1 * B2[2]);

        var d = [x1 - x0, y1 - y0, z1 - z0];
        var w = DNA.RW * rs * FR.tap;

        /* quad 1: width along the axis */
        var w1 = [T2[0] * w, T2[1] * w, T2[2] * w];
        /* quad 2: width perpendicular to both the rung and the axis */
        var w2 = normalize3([d[1] * T2[2] - d[2] * T2[1], d[2] * T2[0] - d[0] * T2[2], d[0] * T2[1] - d[1] * T2[0]]);
        w2[0] *= w; w2[1] *= w; w2[2] *= w;

        var n1 = normalize3([d[1] * w1[2] - d[2] * w1[1], d[2] * w1[0] - d[0] * w1[2], d[0] * w1[1] - d[1] * w1[0]]);
        var n2 = normalize3([d[1] * w2[2] - d[2] * w2[1], d[2] * w2[0] - d[0] * w2[2], d[0] * w2[1] - d[1] * w2[0]]);

        gradient(uj, _col2);
        _col2[0] = _col2[0] * 0.85 + 0.08; _col2[1] *= 0.7; _col2[2] = _col2[2] * 0.9 + 0.04;

        put(x0 - w1[0], y0 - w1[1], z0 - w1[2], n1[0], n1[1], n1[2], _col2);
        put(x0 + w1[0], y0 + w1[1], z0 + w1[2], n1[0], n1[1], n1[2], _col2);
        put(x1 - w1[0], y1 - w1[1], z1 - w1[2], n1[0], n1[1], n1[2], _col2);
        put(x1 + w1[0], y1 + w1[1], z1 + w1[2], n1[0], n1[1], n1[2], _col2);

        put(x0 - w2[0], y0 - w2[1], z0 - w2[2], n2[0], n2[1], n2[2], _col2);
        put(x0 + w2[0], y0 + w2[1], z0 + w2[2], n2[0], n2[1], n2[2], _col2);
        put(x1 - w2[0], y1 - w2[1], z1 - w2[2], n2[0], n2[1], n2[2], _col2);
        put(x1 + w2[0], y1 + w2[1], z1 + w2[2], n2[0], n2[1], n2[2], _col2);
    }
}

/* ------------------------------------------------------------------ */
/*  Dust and blood cells                                               */
/* ------------------------------------------------------------------ */
var dustData = [];
for (var q = 0; q < 300; q++) {
    dustData.push((Math.random() - 0.5) * 24, -Math.random() * 36 + 4, (Math.random() - 0.5) * 24);
}
var dustBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, dustBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(dustData), gl.STATIC_DRAW);

var cells = [];
for (var n = 0; n < 70; n++) {
    cells.push({
        x: (Math.random() - 0.5) * 14,
        y: (Math.random() - 0.5) * 18,
        z: -12 + Math.random() * 13,
        tilt: Math.random() * 6,
        spin: Math.random() * 6,
        speed: 0.4 + Math.random() * 0.8
    });
}

var CAP_TOP  = mul(translate(0,  1.5,  0), scale(0.7, 0.7, 0.7));
var CAP_BOT  = mul(translate(0, -1.5,  0), scale(0.7, 0.7, 0.7));
var CYL_TOP  = mul(translate(0,  0.75, 0), scale(0.7, 1.5, 0.7));
var CYL_BOT  = mul(translate(0, -0.75, 0), scale(0.7, 1.5, 0.7));
var IDENTITY = identity();

/* ------------------------------------------------------------------ */
/*  Canvas size + DOM hooks                                            */
/* ------------------------------------------------------------------ */
var W, H, dpr;
function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width  = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
resize();

var copies = document.querySelectorAll('.copy');
var bar    = document.querySelector('.bar');
var header = document.querySelector('header');
var mainEl = document.querySelector('main');

function scrollProgress() {
    var max = mainEl.offsetHeight - window.innerHeight;
    return max > 0 ? clamp(window.scrollY / max, 0, 1) : 0;
}

function updateText() {
    copies.forEach(function (el) {
        var r = el.getBoundingClientRect();
        var d = Math.abs((r.top + r.bottom) / 2 - H / 2) / (H * 0.55);
        el.style.setProperty('--o', Math.max(0, 1 - d * d).toFixed(3));
    });
    bar.style.setProperty('--p', scrollProgress().toFixed(3));
    header.classList.toggle('solid', window.scrollY > 40);
}
window.addEventListener('scroll', updateText, { passive: true });
window.addEventListener('resize', updateText);
updateText();

/* ------------------------------------------------------------------ */
/*  Render loop                                                        */
/* ------------------------------------------------------------------ */
gl.enable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
gl.clearColor(0.051, 0.102, 0.165, 1);

var CORAL = [0.91, 0.38, 0.30], SALINE = [0.86, 0.93, 0.95];
var smoothP = 0;

function bindMesh(m) {
    gl.bindBuffer(gl.ARRAY_BUFFER, m.vbo);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ibo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
}

function drawMesh(m, model, color) {
    gl.uniformMatrix4fv(L.uModel, false, model);
    gl.uniform3fv(L.uColor, color);
    gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
}

function frame(time) {
    requestAnimationFrame(frame);
    var t = reduceMotion ? 0 : time / 1000;

    smoothP += (scrollProgress() - smoothP) * (reduceMotion ? 1 : 0.06);
    var p = smoothP, mobile = W < 700 || W / H <= 1;

    var cam = {
        x: Math.sin(p * 5) * 1.6,
        y: -p * 28,
        z: (mobile ? 12 : 9) - 1.2 * smoothstep(0.62, 1, p)
    };

    var fov  = mobile ? 65 : 50;
    var proj = perspective(fov, W / H, 0.1, 100);
    if (mobile) proj[9] = -0.42;
    var view = translate(-cam.x, -cam.y, -cam.z);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    var aspect    = W / H;
    var halfWidth = Math.tan(fov * Math.PI / 360) * 9 * aspect;
    var sideX     = mobile ? 0 : halfWidth * 0.55;
    var fit       = clamp(aspect / 1.7, 0.55, 1);
    var rs        = mobile ? Math.min(fit, 0.6) : fit;
    var capsuleX  = -sideX;

    /* ---- DNA ribbons ---- */
    dnaCX  = sideX;
    dnaAmp = halfWidth * (mobile ? 0.3 : 0.22);
    buildDNA(0, p, rs);          // time = 0: the DNA only moves when you scroll

    gl.useProgram(dnaProg);
    gl.uniformMatrix4fv(D.uProj, false, proj);
    gl.uniformMatrix4fv(D.uView, false, view);
    gl.uniform3f(D.uCam, cam.x, cam.y, cam.z);
    gl.uniform3f(D.uKeyPos, 6, cam.y + 4, 8);
    gl.uniform3f(D.uRimPos, -8, cam.y - 2, -6);

    gl.bindBuffer(gl.ARRAY_BUFFER, dnaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, dnaVerts, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, dnaIbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24);
    gl.drawElements(gl.TRIANGLES, dnaCount, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(2);

    /* ---- capsule + cells ---- */
    gl.useProgram(litProg);
    gl.uniformMatrix4fv(L.uProj, false, proj);
    gl.uniformMatrix4fv(L.uView, false, view);
    gl.uniform3f(L.uCam, cam.x, cam.y, cam.z);
    gl.uniform3f(L.uKeyPos, 6, cam.y + 4, 8);
    gl.uniform3f(L.uRimPos, -8, cam.y - 2, -6);

    var tilt = 0.4 + Math.sin(t * 0.6) * 0.1;
    var pill = mul(translate(capsuleX, PILL_Y, 0), rotateZ(p * 9), rotateX(tilt), scale(1.3 * fit, 1.3 * fit, 1.3 * fit));

    bindMesh(sphere);
    gl.uniformMatrix4fv(L.uParent, false, pill);
    drawMesh(sphere, CAP_TOP, CORAL);
    drawMesh(sphere, CAP_BOT, SALINE);

    gl.uniformMatrix4fv(L.uParent, false, IDENTITY);
    cells.forEach(function (c) {
        if (!reduceMotion) {
            c.tilt += 0.01 * c.speed;
            c.spin += 0.004 * c.speed;
            c.y    += 0.01 * c.speed;
            if (c.y > 9) c.y = -9;
        }
        drawMesh(
            sphere,
            mul(translate(c.x + sideX * 0.5, c.y - 28, c.z), rotateZ(c.spin), rotateX(c.tilt), scale(0.4, 0.15, 0.4)),
            CORAL
        );
    });

    bindMesh(cylinder);
    gl.uniformMatrix4fv(L.uParent, false, pill);
    drawMesh(cylinder, CYL_TOP, CORAL);
    drawMesh(cylinder, CYL_BOT, SALINE);

    /* ---- dust points ---- */
    gl.useProgram(flatProg);
    gl.disableVertexAttribArray(1);
    gl.uniformMatrix4fv(F.uProj, false, proj);
    gl.uniformMatrix4fv(F.uView, false, view);
    gl.uniformMatrix4fv(F.uModel, false, IDENTITY);
    gl.uniform4f(F.uColor, 0.86, 0.93, 0.95, 0.6);
    gl.uniform1f(F.uSize, 2 * dpr);

    gl.bindBuffer(gl.ARRAY_BUFFER, dustBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
    gl.drawArrays(gl.POINTS, 0, dustData.length / 3);
}

requestAnimationFrame(frame);