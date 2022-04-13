/* eslint-disable */
import "babel-polyfill";
import * as tf from "@tensorflow/tfjs";
tf.ENV.set("WEBGL_PACK", false); // This needs to be done otherwise things run very slow v1.0.4
import links from "./links";

class Core {
  constructor() {
    Promise.all([
      this.loadMobileNetStyleModel(),
      this.loadSeparableTransformerModel(),
    ]).then(([styleNet, transformNet]) => {
      console.log("Loaded styleNet");
      this.styleNet = styleNet;
      this.transformNet = transformNet;
    });

    //Checks if bundle is part of CSnap
    this.ide = null;
    if (typeof world !== "undefined") {
      this.ide = world.children[0];
    }
  }

  generateStylizedImage(options) {
    let generic = {
      contentImage: "images/beach.jpg",
      sourceImage: "images/statue_of_liberty.jpg",
      styleModel: "mobilenet",
      transformModel: "separable",
      styleRatio: 0.5,
      contentSize: 100,
      sourceSize: 100,
    };

    if (options) {
      Object.assign(generic, options);
    }

    //TODO Convert to dynamic (width sizing issues prevented this from being done earlier)
    this.contentImg = document.getElementById('base-image')
    this.contentImg.removeAttribute('height');
    this.contentImg.removeAttribute('width');

    this.contentImg.src = generic.contentImage;
    this.contentImg.height = this.contentImg.height * generic.contentSize;
    this.contentImg.width = this.contentImg.width * generic.contentSize;
  
    this.styleImg = document.getElementById('style-image')
    this.styleImg.removeAttribute('height');
    this.styleImg.removeAttribute('width');
    this.styleImg.src = generic.sourceImage;
    this.styleImg.height = this.styleImg.height * generic.sourceSize;
    this.styleImg.width = this.styleImg.width * generic.sourceSize;

    this.styleRatio = generic.styleRatio;
    this.stylized = document.getElementById("style-canvas");


    // Calls the block that loads the progress bar to user
    if (typeof world !== "undefined") {
      let ide = world.children[0];
      ide.broadcast("startProgress");
    }

    Promise.all([this.loadStyleModel(generic.styleModel), this.loadTransformModel(generic.transformModel)]).then(
      ([styleNet, transformNet]) => {
        console.log("Loaded styleNet");
        this.styleNet = styleNet;
        this.transformNet = transformNet;

        this.startStyling().finally(() => {
          let a = document.createElement("a");

          this.fixStylizedImage();
          a.setAttribute("download", "output.png");
          a.setAttribute(
            "href",
            this.stylized
              .toDataURL("image/png", 1.0)
          );
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);

          // Calls the block that hides the progress bar to user
          if (typeof world !== "undefined") {
            let ide = world.children[0];
            ide.broadcast("endProgress");
          }
        });
      }
    );
  }

  // ? Where does tensorflow call to duplicate the canvas by 2?
  // This fixes the doubling with the canvas, producing a proper image
  fixStylizedImage(){
    let canvas = document.getElementById('style-canvas');
    let ctx = canvas.getContext('2d');
    let width = canvas.width/2
    let height = canvas.height/2
    let imageData = ctx.getImageData(0,0, width, height);
    canvas.width = width;
    canvas.height = height;
    ctx.putImageData(imageData, 0, 0);
  }

  async loadStyleModel(style) {
    return style == "inception"
      ? this.loadInceptionStyleModel()
      : this.loadMobileNetStyleModel();
  }

  async loadTransformModel(transform) {
    return transform == "original"
      ? this.loadOriginalTransformerModel()
      : this.loadSeparableTransformerModel();
  }

  async loadMobileNetStyleModel() {
    if (!this.mobileStyleNet) {
      this.mobileStyleNet = await tf.loadGraphModel(
        "/static/csnap_pro/csdt/ast/saved_model_style_js/model.json"
      );
      if (this.ide) {
        this.ide.broadcast("fastModelLoad");
      }
    }

    return this.mobileStyleNet;
  }

  async loadInceptionStyleModel() {
    if (!this.inceptionStyleNet) {
      this.inceptionStyleNet = await tf.loadGraphModel(
        "/static/csnap_pro/csdt/ast/saved_model_style_inception_js/model.json"
      );
      if (this.ide) {
        this.ide.broadcast("highModelLoad");
      }
    }

    return this.inceptionStyleNet;
  }

  async loadOriginalTransformerModel() {
    if (!this.originalTransformNet) {
      this.originalTransformNet = await tf.loadGraphModel(
        "/static/csnap_pro/csdt/ast/saved_model_transformer_js/model.json"
      );
      if (this.ide) {
        this.ide.broadcast("highTransformLoad");
      }
    }

    return this.originalTransformNet;
  }

  async loadSeparableTransformerModel() {
    if (!this.separableTransformNet) {
      this.separableTransformNet = await tf.loadGraphModel(
        "/static/csnap_pro/csdt/ast/saved_model_transformer_separable_js/model.json"
      );
      if (this.ide) {
        this.ide.broadcast("fastTransformLoad");
      }
    }

    return this.separableTransformNet;
  }

  async startStyling() {
    await tf.nextFrame();
    console.log("Generating 100D style representation");
    await tf.nextFrame();
    let bottleneck = await tf.tidy(() => {
      return this.styleNet.predict(
        tf.browser
          .fromPixels(this.styleImg)
          .toFloat()
          .div(tf.scalar(255))
          .expandDims()
      );
    });
    if (this.styleRatio !== 1.0) {
      console.log("Generating 100D identity style representation")
      await tf.nextFrame();
      const identityBottleneck = await tf.tidy(() => {
        return this.styleNet.predict(
          tf.browser
            .fromPixels(this.contentImg)
            .toFloat()
            .div(tf.scalar(255))
            .expandDims()
        );
      });
      const styleBottleneck = bottleneck;
      bottleneck = await tf.tidy(() => {
        const styleBottleneckScaled = styleBottleneck.mul(
          tf.scalar(this.styleRatio)
        );
        const identityBottleneckScaled = identityBottleneck.mul(
          tf.scalar(1.0 - this.styleRatio)
        );
        return styleBottleneckScaled.addStrict(identityBottleneckScaled);
      });
      styleBottleneck.dispose();
      identityBottleneck.dispose();
    }
    console.log("Stylizing image...");
    await tf.nextFrame();
    const stylized = await tf.tidy(() => {
      return this.transformNet
        .predict([
          tf.browser
            .fromPixels(this.contentImg)
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(),
          bottleneck,
        ])
        .squeeze();
    });
    await tf.browser.toPixels(stylized, this.stylized);
    bottleneck.dispose(); // Might wanna keep this around
    stylized.dispose();
  }

  async startCombining() {
    await tf.nextFrame();
    // this.combineButton.textContent =
    //   "Generating 100D style representation of image 1";
    await tf.nextFrame();
    const bottleneck1 = await tf.tidy(() => {
      return this.styleNet.predict(
        tf.browser
          .fromPixels(this.combStyleImg1)
          .toFloat()
          .div(tf.scalar(255))
          .expandDims()
      );
    });

    // this.combineButton.textContent =
    //   "Generating 100D style representation of image 2";
    await tf.nextFrame();
    const bottleneck2 = await tf.tidy(() => {
      return this.styleNet.predict(
        tf.browser
          .fromPixels(this.combStyleImg2)
          .toFloat()
          .div(tf.scalar(255))
          .expandDims()
      );
    });

    // this.combineButton.textContent = "Stylizing image...";
    await tf.nextFrame();
    const combinedBottleneck = await tf.tidy(() => {
      const scaledBottleneck1 = bottleneck1.mul(
        tf.scalar(1 - this.combStyleRatio)
      );
      const scaledBottleneck2 = bottleneck2.mul(tf.scalar(this.combStyleRatio));
      return scaledBottleneck1.addStrict(scaledBottleneck2);
    });

    const stylized = await tf.tidy(() => {
      return this.transformNet
        .predict([
          tf.browser
            .fromPixels(this.combContentImg)
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(),
          combinedBottleneck,
        ])
        .squeeze();
    });
    await tf.browser.toPixels(stylized, this.combStylized);
    bottleneck1.dispose(); // Might wanna keep this around
    bottleneck2.dispose();
    combinedBottleneck.dispose();
    stylized.dispose();
  }

  async benchmark() {
    const x = tf.randomNormal([1, 256, 256, 3]);
    const bottleneck = tf.randomNormal([1, 1, 1, 100]);

    let styleNet = await this.loadInceptionStyleModel();
    let time = await this.benchmarkStyle(x, styleNet);
    styleNet.dispose();

    styleNet = await this.loadMobileNetStyleModel();
    time = await this.benchmarkStyle(x, styleNet);
    styleNet.dispose();

    let transformNet = await this.loadOriginalTransformerModel();
    time = await this.benchmarkTransform(x, bottleneck, transformNet);
    transformNet.dispose();

    transformNet = await this.loadSeparableTransformerModel();
    time = await this.benchmarkTransform(x, bottleneck, transformNet);
    transformNet.dispose();

    x.dispose();
    bottleneck.dispose();
  }

  async benchmarkStyle(x, styleNet) {
    const profile = await tf.profile(() => {
      tf.tidy(() => {
        const dummyOut = styleNet.predict(x);
        dummyOut.print();
      });
    });
    console.log(profile);
    const time = await tf.time(() => {
      tf.tidy(() => {
        for (let i = 0; i < 10; i++) {
          const y = styleNet.predict(x);
          y.print();
        }
      });
    });
    console.log(time);
  }

  async benchmarkTransform(x, bottleneck, transformNet) {
    const profile = await tf.profile(() => {
      tf.tidy(() => {
        const dummyOut = transformNet.predict([x, bottleneck]);
        dummyOut.print();
      });
    });
    console.log(profile);
    const time = await tf.time(() => {
      tf.tidy(() => {
        for (let i = 0; i < 10; i++) {
          const y = transformNet.predict([x, bottleneck]);
          y.print();
        }
      });
    });
    console.log(time);
  }
}

// function validateTextureSize(width, height) {
//   var maxTextureSize = environment_1.ENV.get('WEBGL_MAX_TEXTURE_SIZE');
//   if ((width <= 0) || (height <= 0)) {
//       var requested = "[" + width + "x" + height + "]";
//       if(typeof world !== 'undefined'){
//           let ide = world.children[0]
//           ide.broadcast('sizeError')
//       }
//       throw new Error('Requested texture size ' + requested + ' is invalid.');
//   }
//   if ((width > maxTextureSize) || (height > maxTextureSize)) {
//       var requested = "[" + width + "x" + height + "]";
//       var max = "[" + maxTextureSize + "x" + maxTextureSize + "]";
//       if(typeof world !== 'undefined'){
//           let ide = world.children[0]
//           ide.broadcast('sizeError')
//       }
//       throw new Error('Requested texture size ' + requested +
//           ' greater than WebGL maximum on this browser / GPU ' + max + '.');
//   }
// }

window.application = new Core();
