
class IOState
{
    constructor(canvas)
    {
        // Props
        this.angle = [0, 0];
        this.mousePos = [0, 0];
        this.hasChange = false;
        this.canvas = canvas;
    }

    // References
    OnMouseMove(event)
    {
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

        if(this.hasChange)
            requestAnimationFrame(RenderGL);
    };

    OnMousePress(event)
    {
        if(window.TouchEvent &&
           event instanceof TouchEvent)
        {
            this.mousePos = [event.touches[0].clientX,
                             event.touches[0].clientY]
        }
        else
        {
            this.mousePos = [event.clientX,
                             event.clientY]
        }
        this.hasChange = true;
        event.preventDefault();

        if(this.hasChange)
            requestAnimationFrame(RenderGL);
    };
    OnMouseRelease(event) { };

    OnResize(event)
    {
        this.hasChange = true;
        console.log(`MMove (${width}, ${height})`);
    }
}

// ============ //
//   Statics    //
// ============ //
const PPGenericVert = `#version 300 es

    in vec2 vPos;
    out vec2 fUV;

    void main(void)
    {
        //		Pos					UV
	    //	-1.0f, 3.0f,	-->	0.0f, 2.0f,
	    //	3.0f, -1.0f,	-->	2.0f, 0.0f,
	    //	-1.0f, -1.0f,	-->	0.0f, 0.0f
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
        vec3 inColor =  vec3(inColorUI) / vec3(65535.0);


        // Gamma
        //vec3 outColor = pow(inColor, vec3(0.4545f));
        //fboColor = vec4(outColor , 1.0f);

        fboColor = vec4(1, 0, 0, 1);
    }
`;

const PPTriangle = [3.0, -1.0, -1.0, 3.0, -1.0, -1.0];

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
const MAX_FRAMES = 10;
let frameCount = 0;
let prevWH = [0, 0];

//
function RenderGL(timestamp)
{
    console.log(`Canvas (${canvas.scrollWidth}, ${canvas.scrollHeight})`);

    frameCount++;

    curFBOIndex = juggleIndex;
    prevFBOIndex = (juggleIndex + 1) % 2;

    if(ioState.hasChange ||
        prevWH[0] != canvas.scrollWidth ||
        prevWH[1] != canvas.scrollHeight)
    {
        prevWH = [canvas.scrollWidth, canvas.scrollHeight];
        frameCount = 0;
        ResetFBOs();
        ioState.hasChange = false;
    }

    // =========== //
    // Path Trace! //
    // =========== //
    //gl.bindFramebuffer(gl.FRAMEBUFFER, pathTraceFBOs[curFBOIndex]);
    //gl.viewport(0, 0, ioState.width, ioState.height);
    //gl.useProgram(accumImageShader);
    //// Textures
    //gl.activeTexture(gl.TEXTURE0);
    //gl.bindTexture(gl.TEXTURE_2D, pathTraceTextures[prevFBOIndex]);
    //// Uniforms
    //// TODO: ....
    //gl.drawArrays(gl.GL_TRIANGLES, 0, 3);

    // ============ //
    // TM & Display //
    // ============ //
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, ioState.width, ioState.height);
    gl.useProgram(accumImageShader);
    // Textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pathTraceTextures[curFBOIndex]);
    gl.drawArrays(gl.GL_TRIANGLES, 0, 3);

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

        gl.clearBufferuiv(gl.COLOR, 0, [0, 0, 0, 0]);
    }
}

function InitGL()
{
    canvas = document.getElementById("glcanvas");
    gl = canvas.getContext("webgl2",
    {
        antialias: true,
        depth: false,
        stencil: false,
        alpha: false,
        failIfMajorPerformanceCaveat: true,
    });
    ioState = new IOState(canvas);

    const image = document.getElementById("staticImg");
    //
    //canvas.width = 200;
    //canvas.height = 200;
    canvas.style.display = "none";

    canvas.parentElement
    //image.style.display = "none";

    //const width = canvas.scrollWidth;
    //const height = canvas.scrollHeight;

    // Add Listeners
    canvas.addEventListener("mousedown", (event) => ioState.OnMousePress(event));
    canvas.addEventListener("mouseup", (event) => ioState.OnMouseRelease(event));
    canvas.addEventListener("mousemove", (event) => ioState.OnMouseMove(event));
    //
    canvas.addEventListener("touchstart", (event) => ioState.OnMousePress(event));
    canvas.addEventListener("touchend", (event) => ioState.OnMouseRelease(event));
    canvas.addEventListener("touchmove", (event) => ioState.OnMouseMove(event));

    canvas.addEventListener("resize", (event) => ioState.OnResize(event));
    //canvas.parentElement.addEventListener("resize", (event) => ioState.OnResize(event));

    if(!gl || !(gl instanceof WebGL2RenderingContext))
    {
        //image.style.display = "initial";
        return;
    }

    // Create the shaders
    accumImageShader = CreateProgramGL(PPGenericVert, AccumImgFrag);
    //pathTraceShader = CreateProgramGL(PPGenericVert, FetchShader('./_static/cBox.frag'))

    // Create background FBO for path tracing
    pathTraceFBOs = [gl.createFramebuffer(), gl.createFramebuffer()];
    pathTraceTextures = [gl.createTexture(), gl.createTexture()];
    //ResetFBOs();

    //
    juggleIndex = 0;

    // We do not use depth testing
    gl.disable(gl.DEPTH_TEST);


    canvas.style.display = "block";
    //canvas.style.width = '100%';
    //canvas.style.height = '100%';
    //// ...then set the internal size to match
    //canvas.width = canvas.offsetWidth;
    //canvas.height = canvas.offsetHeight;

    requestAnimationFrame(RenderGL);
}

async function FetchShader(fName)
{
    const response = await fetch(fName);
    if(!response.ok)
    {
        throw new Error(`Response status: ${response.status}`);
    }
    const text = await response.text();
    return text;
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
            //console.log("CompilerError \n" + info);
            console.log("CompilerError \n" + sourceCode + "\n" + info);

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