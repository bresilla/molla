// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="introduction.html">Introduction</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Foundations</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="architecture/overview.html"><strong aria-hidden="true">1.</strong> Architecture overview</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="architecture/workspace.html"><strong aria-hidden="true">1.1.</strong> Workspace layout</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="architecture/crates.html"><strong aria-hidden="true">1.2.</strong> Crate boundaries</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="architecture/features.html"><strong aria-hidden="true">1.3.</strong> Feature flags</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="architecture/build.html"><strong aria-hidden="true">1.4.</strong> Build &amp; test commands</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="architecture/design_rules.html"><strong aria-hidden="true">1.5.</strong> Design rules</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Math &amp; Numerics</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/types.html"><strong aria-hidden="true">2.</strong> Vectors, matrices, transforms</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/spatial.html"><strong aria-hidden="true">3.</strong> Spatial algebra</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/dense3.html"><strong aria-hidden="true">4.</strong> Small-matrix decompositions</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/sparse.html"><strong aria-hidden="true">5.</strong> Sparse linear algebra</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/bsr_cpu.html"><strong aria-hidden="true">5.1.</strong> BsrMatrix (CPU)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/bsr_gpu.html"><strong aria-hidden="true">5.2.</strong> BsrGpu (GPU)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/spgemm.html"><strong aria-hidden="true">5.3.</strong> SpGEMM design</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/iterative.html"><strong aria-hidden="true">6.</strong> Iterative solvers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/optim.html"><strong aria-hidden="true">7.</strong> Optimizers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/coloring.html"><strong aria-hidden="true">8.</strong> Graph coloring</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="math/scan_sort.html"><strong aria-hidden="true">9.</strong> Scan &amp; 64-bit radix sort</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Compute &amp; GPU</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="compute/runtime.html"><strong aria-hidden="true">10.</strong> GPU runtime</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="compute/storage.html"><strong aria-hidden="true">11.</strong> Storage abstractions</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="compute/kernels.html"><strong aria-hidden="true">12.</strong> Kernel infrastructure</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="compute/tile.html"><strong aria-hidden="true">13.</strong> Tile compute primitives</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="compute/persistent.html"><strong aria-hidden="true">14.</strong> Persistent-buffer pattern</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Geometry &amp; Collision</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/shapes.html"><strong aria-hidden="true">15.</strong> Shape primitives</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/hull.html"><strong aria-hidden="true">16.</strong> Convex hull builder</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/bvh.html"><strong aria-hidden="true">17.</strong> BVH</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/spatial_hash.html"><strong aria-hidden="true">18.</strong> Spatial hash</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/broad_phase.html"><strong aria-hidden="true">19.</strong> Broad phase</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/narrow_phase.html"><strong aria-hidden="true">20.</strong> Narrow phase</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/gjk.html"><strong aria-hidden="true">20.1.</strong> GJK</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/epa.html"><strong aria-hidden="true">20.2.</strong> EPA</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/mpr.html"><strong aria-hidden="true">20.3.</strong> MPR</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/contact_reduction.html"><strong aria-hidden="true">21.</strong> Contact reduction</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/sparse_grid.html"><strong aria-hidden="true">22.</strong> Sparse grids (NanoVDB-equivalent)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/sdf.html"><strong aria-hidden="true">23.</strong> Marching cubes &amp; SDF baking</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/decimate.html"><strong aria-hidden="true">24.</strong> Mesh decimation</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/gaussian_splat.html"><strong aria-hidden="true">25.</strong> Gaussian splats</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/soft_contact.html"><strong aria-hidden="true">26.</strong> Soft contact pipeline</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="geometry/hydroelastic.html"><strong aria-hidden="true">27.</strong> Hydroelastic contact</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Simulation Core</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/model.html"><strong aria-hidden="true">28.</strong> Data model</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/builder.html"><strong aria-hidden="true">29.</strong> ModelBuilder</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/joints.html"><strong aria-hidden="true">30.</strong> Joints</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/joints_basic.html"><strong aria-hidden="true">30.1.</strong> Revolute / Prismatic / Fixed / Free</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/joints_multi_dof.html"><strong aria-hidden="true">30.2.</strong> Ball / D6 / Distance</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/joints_cable.html"><strong aria-hidden="true">30.3.</strong> CABLE joint</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/joints_mimic.html"><strong aria-hidden="true">30.4.</strong> Joint mimic</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/featherstone.html"><strong aria-hidden="true">31.</strong> Articulation (Featherstone ABA)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/spatial_kernels.html"><strong aria-hidden="true">32.</strong> CRBA / RNEA / Jacobian</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sim/ik.html"><strong aria-hidden="true">33.</strong> Inverse kinematics</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Solvers</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/overview.html"><strong aria-hidden="true">34.</strong> Solvers overview</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/semi_implicit.html"><strong aria-hidden="true">35.</strong> Semi-implicit integrator</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/xpbd.html"><strong aria-hidden="true">36.</strong> XPBD</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/vbd.html"><strong aria-hidden="true">37.</strong> VBD</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/mpm.html"><strong aria-hidden="true">38.</strong> MPM</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/mpm_explicit.html"><strong aria-hidden="true">38.1.</strong> Explicit</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/mpm_semi_implicit.html"><strong aria-hidden="true">38.2.</strong> Semi-implicit</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/mpm_implicit.html"><strong aria-hidden="true">38.3.</strong> Implicit Bingham</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="solvers/barrier_newton.html"><strong aria-hidden="true">39.</strong> Barrier-Newton contact (IPC)</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Finite-Element Method</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/overview.html"><strong aria-hidden="true">40.</strong> FEM overview</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/mesh.html"><strong aria-hidden="true">41.</strong> Mesh &amp; function spaces</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/quadrature.html"><strong aria-hidden="true">42.</strong> Quadrature rules</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/elements.html"><strong aria-hidden="true">43.</strong> Element matrices</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/assembly.html"><strong aria-hidden="true">44.</strong> Assembly</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/bc.html"><strong aria-hidden="true">45.</strong> Boundary conditions</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/solvers.html"><strong aria-hidden="true">46.</strong> Solvers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/p2.html"><strong aria-hidden="true">47.</strong> Higher-order elements (P2)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/dg.html"><strong aria-hidden="true">48.</strong> Discontinuous Galerkin</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/stokes_ns.html"><strong aria-hidden="true">49.</strong> Stokes &amp; Navier-Stokes</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/mixed_elasticity.html"><strong aria-hidden="true">50.</strong> Mixed (u, p) elasticity</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/apic.html"><strong aria-hidden="true">51.</strong> APIC FEM-MPM coupling</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/adjoint.html"><strong aria-hidden="true">52.</strong> Adjoint / differentiable FEM</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/amr.html"><strong aria-hidden="true">53.</strong> Adaptive mesh refinement (AMR)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/elasticity_gpu.html"><strong aria-hidden="true">54.</strong> GPU FEM elasticity</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/magneto.html"><strong aria-hidden="true">55.</strong> Magnetostatics — H(curl) Nedelec</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/eddy.html"><strong aria-hidden="true">56.</strong> Eddy currents (time-harmonic Maxwell)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/multigrid.html"><strong aria-hidden="true">57.</strong> Multigrid (geometric, AMG, AMS)</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/nonconforming.html"><strong aria-hidden="true">58.</strong> Nonconforming meshes</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/signorini.html"><strong aria-hidden="true">59.</strong> Signorini hard contact</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/friction.html"><strong aria-hidden="true">60.</strong> Coulomb friction</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="fem/phase_field.html"><strong aria-hidden="true">61.</strong> Phase-field fracture</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Sensors</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sensors/raycaster.html"><strong aria-hidden="true">62.</strong> Raycaster</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sensors/tiled_camera.html"><strong aria-hidden="true">63.</strong> TiledCamera</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sensors/imu.html"><strong aria-hidden="true">64.</strong> IMU</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sensors/contact_sensor.html"><strong aria-hidden="true">65.</strong> Contact sensor</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="sensors/frame_transform.html"><strong aria-hidden="true">66.</strong> Frame transform</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Actuators &amp; Control</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="actuators/pipeline.html"><strong aria-hidden="true">67.</strong> Composable actuator pipeline</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="actuators/pd_pid.html"><strong aria-hidden="true">68.</strong> PD / PID controllers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="actuators/neural.html"><strong aria-hidden="true">69.</strong> Neural controllers</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="actuators/dc_motor.html"><strong aria-hidden="true">70.</strong> DC motor clamping</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="actuators/muscle.html"><strong aria-hidden="true">71.</strong> Muscle (Hill-type + SEE)</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Vehicle</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="vehicle/overview.html"><strong aria-hidden="true">72.</strong> molla-vehicle overview</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Differentiable Physics</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/tape.html"><strong aria-hidden="true">73.</strong> Tape-based autograd</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/featherstone.html"><strong aria-hidden="true">74.</strong> Featherstone reverse adjoint</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/xpbd_vbd.html"><strong aria-hidden="true">75.</strong> XPBD / VBD adjoint</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/mpm.html"><strong aria-hidden="true">76.</strong> MPM adjoint</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/fem.html"><strong aria-hidden="true">77.</strong> FEM adjoint</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/raycast.html"><strong aria-hidden="true">78.</strong> Raycast adjoint</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/splat.html"><strong aria-hidden="true">79.</strong> Differentiable Gaussian splat renderer</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/cloth_softbody_gpu.html"><strong aria-hidden="true">80.</strong> GPU cloth &amp; softbody adjoints</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="autograd/stam_fluid.html"><strong aria-hidden="true">81.</strong> Eulerian fluid adjoint</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Viewer</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="adapters/bevy.html"><strong aria-hidden="true">82.</strong> Bevy adapter</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="viewer/overview.html"><strong aria-hidden="true">83.</strong> Viewer overview</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="viewer/scenes.html"><strong aria-hidden="true">84.</strong> Scene catalog (#1-124)</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Testing &amp; Performance</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/strategy.html"><strong aria-hidden="true">85.</strong> Test strategy &amp; Newton oracle</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/parity.html"><strong aria-hidden="true">86.</strong> CPU/GPU parity testing</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/fd_gradients.html"><strong aria-hidden="true">87.</strong> Finite-difference gradient checks</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/benchmarks.html"><strong aria-hidden="true">88.</strong> Benchmarks</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="testing/gpu_status.html"><strong aria-hidden="true">89.</strong> GPU port status matrix</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><li class="part-title">Roadmap</li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="roadmap/stages.html"><strong aria-hidden="true">90.</strong> Stage history</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="roadmap/open.html"><strong aria-hidden="true">91.</strong> Open items</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="roadmap/oos.html"><strong aria-hidden="true">92.</strong> Out-of-scope</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="about.html">About this book</a></span></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            // Check both with and without the '.html' suffix to be robust against pretty URLs
            if (link.href.replace(/\.html$/, '') === current_page.replace(/\.html$/, '')
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level > level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

