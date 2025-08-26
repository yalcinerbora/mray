
class IOState
{
    constructor(canvas, camPos, camUp)
    {
        // Props
        this.angle = [0, 0];
        this.mousePos = [0, 0];
        this.hasChange = false;
        this.canvas = canvas;
        this.clicked = false;
        this.initCamPos = camPos;
        this.camPos = camPos;
        this.camUp = camUp;
    }

    // References
    OnMouseMove(event)
    {
        if(!this.clicked) return;

        var cur = [0, 0];
        if(window.TouchEvent && event instanceof TouchEvent)
        {
            cur = [event.touches[0].clientX,
                   event.touches[0].clientY];
        }
        else
        {
            cur = [event.clientX, event.clientY];
        }
        // Set angle by yourself
        let diffX = (cur[0] - this.mousePos[0]);
        let diffY = (cur[1] - this.mousePos[1]);
        this.angle[0] += diffX;
        this.angle[1] += diffY;
        this.mousePos = cur;

        this.hasChange = (diffX != 0 || diffY != 0);

        this.UpdateCamPos();

        if(this.hasChange)
            requestAnimationFrame(RenderGL);
    };

    OnMousePress(event)
    {
        this.clicked = true;

        if(window.TouchEvent &&
           event instanceof TouchEvent)
        {
            this.mousePos = [event.touches[0].clientX,
                             event.touches[0].clientY];
        }
        else
        {
            this.mousePos = [event.clientX,
                             event.clientY];
        }
        this.hasChange = true;
    };

    OnMouseRelease(event)
    {
        this.clicked = false;
    };

    OnResize(event)
    {
        this.hasChange = true;
        requestAnimationFrame(RenderGL);
    }

    UpdateCamPos()
    {
        let len = this.initCamPos[0] * this.initCamPos[0] +
                  this.initCamPos[1] * this.initCamPos[1] +
                  this.initCamPos[2] * this.initCamPos[2];
        len = Math.sqrt(len);

        let sX = Math.sin(-this.angle[1] * Math.PI * 0.005555);
        let cX = Math.cos(-this.angle[1] * Math.PI * 0.005555);
        let sY = Math.sin(-this.angle[0] * Math.PI * 0.005555);
        let cY = Math.cos(-this.angle[0] * Math.PI * 0.005555);
        // This is the rotation matrix X and rotation matrix Y (with the angles above)
        // multiplied by Z axis (initially cam is towards Z)
        let vRot = [sY * cX, -sX, cX * cY];
        vRot[0] *= len;
        vRot[1] *= len;
        vRot[2] *= len;

        // Translate it back (gaze is [0,1,0])
        vRot[1] += 1;
        //
        this.camPos = vRot;

        // Realign the up vector
        const x = Math.abs(Math.floor((this.angle[1] - 90.0) / 180));
        if(x % 2 == 1)
            this.camUp = [0, 1.0, 0];
        else
            this.camUp = [0, -1.0, 0];
    }
}

// ============ //
//   Statics    //
// ============ //
const V_POS_LOC = 0;

const PPGenericVert = `#version 300 es

    layout(location = ${V_POS_LOC})
    in vec2 vPos;

    out vec2 fUV;

    void main(void)
    {
        //      Pos                 UV
        //  -1.0f, 3.0f,    --> 0.0f, 2.0f,
        //  3.0f, -1.0f,    --> 2.0f, 0.0f,
        //  -1.0f, -1.0f,   --> 0.0f, 0.0f
        fUV = (vPos + 1.0f) * 0.5f;
        gl_Position = vec4(vPos, 0.0f, 1.0f);
    }
`;

const AccumImgFrag = `#version 300 es

    precision highp float;

    // In
    in vec2 fUV;

    // Uniforms
    uniform highp usampler2D tRadiance;

    // Out
    out vec4 fboColor;

    void main(void)
    {
        // rgb is rgb, alpha is the sample count
        uvec3 inColorUI = texture(tRadiance, fUV).rgb;
        vec3 inColor =  vec3(inColorUI) * vec3(1.525902189669642e-5);

        // Gamma
        vec3 outColor = pow(inColor, vec3(0.4545f));
        //vec3 outColor = inColor;
        fboColor = vec4(outColor, 1.0f);
    }
`;

const PPTriangle = new Float32Array([3.0, -1.0, -1.0, 3.0, -1.0, -1.0]);

// ============ //
//   Globals    //
// ============ //
// OGL Related
// Init GL
let canvas;
let gl;
let accumImageShader;
let pathTraceShader;
let juggleIndex;
//
let pathTraceFBOs;
let pathTraceTextures;
// Callback Related
let ioState;
// Misc.
const INITIAL_CAM_POS = [0, 1, 6.8];
const INITIAL_CAM_UP = [0, 1, 0];
const MAX_FRAMES = 512;
let frameCount = 0;
let prevWH = [0, 0];

//
function RenderGL(timestamp)
{
    //console.log(`CanvasScroll (${canvas.scrollWidth}, ${canvas.scrollHeight})`);
    //console.log(`Canvas (${canvas.width}, ${canvas.height})`);

    curFBOIndex = juggleIndex;
    prevFBOIndex = (juggleIndex + 1) % 2;

    let fboChanged = prevWH[0] != canvas.scrollWidth ||
                     prevWH[1] != canvas.scrollHeight;
    if(ioState.hasChange || fboChanged)
    {
        if(fboChanged) ResetFBOs();

        prevWH = [canvas.scrollWidth, canvas.scrollHeight];
        frameCount = 0;
        ioState.hasChange = false;
        for(var i = 0; i < 2; i++)
        {
            gl.bindFramebuffer(gl.FRAMEBUFFER, pathTraceFBOs[i]);
            gl.clearBufferuiv(gl.COLOR, 0, [0, 0, 0, 0]);
        }
    }
    frameCount++;

    // =========== //
    // Path Trace! //
    // =========== //
    gl.bindFramebuffer(gl.FRAMEBUFFER, pathTraceFBOs[curFBOIndex]);
    gl.viewport(0, 0, prevWH[0], prevWH[1]);
    gl.useProgram(pathTraceShader);
    // Textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pathTraceTextures[prevFBOIndex]);
    // Uniforms
    const bgColor = GetInheritedBackgroundColor(canvas);
    bgColor[0] = Math.pow(bgColor[0], 2.2);
    bgColor[1] = Math.pow(bgColor[1], 2.2);
    bgColor[2] = Math.pow(bgColor[2], 2.2);

    BindUniforms
    (
        pathTraceShader,
        ["uResolution", "uFrame", "uBGColor", "uCamPos", "uCamUp"],
        (locList) =>
        {
            gl.uniform2f(locList[0], prevWH[0], prevWH[1]);
            gl.uniform1ui(locList[1], frameCount - 1);
            gl.uniform3f(locList[2], bgColor[0], bgColor[1], bgColor[2]);
            gl.uniform3f(locList[3], ioState.camPos[0],
                         ioState.camPos[1], ioState.camPos[2]);
            gl.uniform3f(locList[4], ioState.camUp[0],
                         ioState.camUp[1], ioState.camUp[2]);
        }
    );
    // Draw!
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // ============ //

    // ============ //
    // TM & Display //
    // ============ //
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, prevWH[0], prevWH[1]);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.8, 0.9, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(accumImageShader);
    // Textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pathTraceTextures[curFBOIndex]);
    // Draw!
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // ============ //

    // Change state
    juggleIndex = prevFBOIndex;

    // Continue rendering
    if(frameCount < MAX_FRAMES)
        requestAnimationFrame(RenderGL);
}

function ResetFBOs()
{
    for(var i = 0; i < 2; i++)
    {
        gl.bindFramebuffer(gl.FRAMEBUFFER, pathTraceFBOs[i]);
        gl.bindTexture(gl.TEXTURE_2D, pathTraceTextures[i]);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // We utilize driver's capabilities to auto change the texture
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16UI, canvas.scrollWidth, canvas.scrollWidth, 0,
                      gl.RGBA_INTEGER, gl.UNSIGNED_SHORT, null);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pathTraceTextures[i], 0);
    }
}

async function InitGL()
{
    canvas = document.getElementById("glcanvas");
    gl = canvas.getContext("webgl2",
    {
        antialias   : true,
        depth       : false,
        stencil     : false,
        alpha       : false,
        failIfMajorPerformanceCaveat: true,
    });
    ioState = new IOState(canvas, INITIAL_CAM_POS, INITIAL_CAM_UP);

    // Do not show the canvas untill all shaders are loaded
    canvas.style.display = "none";

    // Add Listeners
    canvas.addEventListener("mousedown", (event) => ioState.OnMousePress(event));
    canvas.addEventListener("mouseup", (event) => ioState.OnMouseRelease(event));
    window.addEventListener("mouseup", (event) => ioState.OnMouseRelease(event));
    window.addEventListener("mousemove", (event) => ioState.OnMouseMove(event));
    //
    const passive = { passive: true };
    canvas.addEventListener("touchstart", (event) => ioState.OnMousePress(event), passive);
    canvas.addEventListener("touchend", (event) => ioState.OnMouseRelease(event));
    window.addEventListener("touchend", (event) => ioState.OnMouseRelease(event));
    window.addEventListener("touchmove", (event) => ioState.OnMouseMove(event), passive);
    //
    window.addEventListener("resize", (event) => ioState.OnResize(event));
    //
    if(!gl || !(gl instanceof WebGL2RenderingContext))
    {
        return;
    }


    // Create Post process vertex buffer
    var vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, PPTriangle, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(V_POS_LOC);
    gl.vertexAttribPointer(V_POS_LOC, 2, gl.FLOAT, false, 0, 0);

    // Create the shaders
    // TODO: We async fetch the shader to make it flexible
    // In future we may load shaders specific to the used system (mobile/pc etc.)
    let ptShaderSource = await FetchShader('_static/cBox.frag');
    accumImageShader = CreateProgramGL(PPGenericVert, AccumImgFrag);
    pathTraceShader = CreateProgramGL(PPGenericVert, ptShaderSource);

    // Create background FBO for path tracing
    pathTraceFBOs = [gl.createFramebuffer(), gl.createFramebuffer()];
    pathTraceTextures = [gl.createTexture(), gl.createTexture()];
    //
    juggleIndex = 0;

    // We do not use depth testing
    gl.disable(gl.DEPTH_TEST);
    canvas.style.display = "block";
    const image = document.getElementById("staticImg");
    image.style.display = "none";

    // From here:
    // https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/light-dark.html
    var observer = new MutationObserver(function (mutations)
    {
        ioState.hasChange = true;
        requestAnimationFrame(RenderGL);
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    //
    requestAnimationFrame(RenderGL);
}

async function FetchShader(fName)
{
    function DecorateWithPrefix(fName)
    {
        // Use this sphinx stuff for relative fetch
        const htmlTag = document.documentElement;
        return (htmlTag.getAttribute('data-content_root') || '/') + fName;
    }

    const absFName = DecorateWithPrefix(fName)
    const response = await fetch(absFName);
    if(!response.ok)
    {
        throw new Error(`Response status: ${response.status}`);
    }
    const text = await response.text();
    return text;
}

// From here:
// https://stackoverflow.com/questions/46336002/how-to-get-computed-background-color-style-inherited-from-parent-element
function GetInheritedBackgroundColor(el)
{
    const RGBToZeroOne = (rgb) =>
    {
        let arr = rgb.slice(rgb.indexOf("(") + 1, rgb.indexOf(")")).split(",");
        return [parseFloat(arr[0]) / 255.0,
                parseFloat(arr[1]) / 255.0,
                parseFloat(arr[2]) / 255.0];
    };
    const GetDefaultBackground = () =>
    {
        // have to add to the document in order to use getComputedStyle
        var div = document.createElement("div");
        document.head.appendChild(div);
        var bg = window.getComputedStyle(div).backgroundColor;
        document.head.removeChild(div);
        return bg;
    };

    // get default style for current browser
    var defaultStyle = GetDefaultBackground(); // typically "rgba(0, 0, 0, 0)"
    // get computed color for el
    var backgroundColor = window.getComputedStyle(el).backgroundColor;
    // if we got a real value, return it
    if(backgroundColor != defaultStyle) return RGBToZeroOne(backgroundColor);
    // if we've reached the top parent el without getting an explicit color, return default
    if(!el.parentElement) return RGBToZeroOne(defaultStyle);
    // otherwise, recurse and try again on parent element
    return GetInheritedBackgroundColor(el.parentElement);
}

function BindUniforms(shader, varNameList, BindFunc)
{
    let locList = [];
    for(var i = 0; i < varNameList.length; i++)
    {
        let loc = gl.getUniformLocation(shader, varNameList[i]);
        locList.push(loc);
    }
    BindFunc(locList);
}

function CreateProgramGL(vShaderSource, fShaderSource)
{
    const CreateShader = function (sourceCode, type)
    {
        // Compiles either a shader of type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
        var shader = gl.createShader(type);
        gl.shaderSource(shader, sourceCode);
        gl.compileShader(shader);
        if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        {
            var info = gl.getShaderInfoLog(shader);
            console.log("CompileError \n" + info);
        }
        return shader;
    };

    let vert = CreateShader(vShaderSource, gl.VERTEX_SHADER);
    let frag = CreateShader(fShaderSource, gl.FRAGMENT_SHADER);
    var program = gl.createProgram();
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);

    if(!gl.getProgramParameter(program, gl.LINK_STATUS))
    {
        var info = gl.getProgramInfoLog(program);
        console.log("LinkError: \n\n" + info);
    }
    return program;
}