(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});

    function setActiveTab(button) {
        document.querySelectorAll(".terminal-tab").forEach((item) => {
            item.classList.toggle("is-active", item === button);
        });
    }

    function activateTab(tabId) {
        const tabInput = document.getElementById("dashboard-tab-input");
        if (!tabInput) {
            return;
        }
        const button = document.querySelector('.terminal-tab[data-tab="' + tabId + '"]');
        tabInput.value = tabId;
        if (button) {
            setActiveTab(button);
        }
        document.body.dispatchEvent(new Event("dashboard-tab-change"));
    }

    function wireTabs(root) {
        const tabInput = document.getElementById("dashboard-tab-input");
        if (!tabInput) {
            return;
        }

        root.querySelectorAll(".terminal-tab").forEach((button) => {
            if (button.dataset.terminalTabBound === "true") {
                return;
            }
            button.dataset.terminalTabBound = "true";
            button.addEventListener("click", function () {
                activateTab(button.dataset.tab || "");
            });
        });
    }

    function wireTabJumpButtons(root) {
        root.querySelectorAll("[data-tab-jump]").forEach((button) => {
            if (button.dataset.terminalTabJumpBound === "true") {
                return;
            }
            button.dataset.terminalTabJumpBound = "true";
            button.addEventListener("click", function () {
                activateTab(button.dataset.tabJump || "");
            });
        });
    }

    function wireTableSort(root) {
        root.querySelectorAll(".terminal-table-sort[data-sort-column]").forEach((button) => {
            if (button.dataset.terminalSortBound === "true") {
                return;
            }
            button.dataset.terminalSortBound = "true";
            button.addEventListener("click", function () {
                const sortColumn = button.dataset.sortColumn || "";
                const sortInputId = button.dataset.sortInputId || "";
                const sortTab = button.dataset.sortTab || "";
                if (!sortColumn || !sortInputId || !sortTab) {
                    return;
                }
                const sortInput = document.getElementById(sortInputId);
                if (!sortInput) {
                    return;
                }
                sortInput.value = sortColumn;
                activateTab(sortTab);
            });
        });
    }

    function wireRiskMetricControls(root) {
        const volSelect = root.querySelector ? root.querySelector("#risk-vol-window-select") : null;
        const sharpeSelect = root.querySelector ? root.querySelector("#risk-sharpe-horizon-select") : null;
        const volInput = document.getElementById("risk-vol-window-days-input");
        const sharpeInput = document.getElementById("risk-sharpe-horizon-input");

        if (volSelect && volInput && volSelect.dataset.terminalRiskVolBound !== "true") {
            volSelect.dataset.terminalRiskVolBound = "true";
            volSelect.addEventListener("change", function () {
                volInput.value = volSelect.value || volInput.value;
                activateTab("risk");
            });
        }

        if (sharpeSelect && sharpeInput && sharpeSelect.dataset.terminalRiskSharpeBound !== "true") {
            sharpeSelect.dataset.terminalRiskSharpeBound = "true";
            sharpeSelect.addEventListener("change", function () {
                sharpeInput.value = sharpeSelect.value || sharpeInput.value;
                activateTab("risk");
            });
        }
    }

    function initResize() {
        const root = document.documentElement;

        const sidebarHandle = document.getElementById("resize-sidebar");
        if (sidebarHandle) {
            let startX = 0;
            let startWidth = 0;
            sidebarHandle.addEventListener("mousedown", function (e) {
                startX = e.clientX;
                startWidth = parseInt(getComputedStyle(root).getPropertyValue("--sidebar-width")) || 320;
                sidebarHandle.classList.add("is-dragging");
                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";

                function onMove(ev) {
                    const newWidth = Math.max(180, Math.min(560, startWidth + ev.clientX - startX));
                    root.style.setProperty("--sidebar-width", newWidth + "px");
                }

                function onUp() {
                    sidebarHandle.classList.remove("is-dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    document.removeEventListener("mousemove", onMove);
                    document.removeEventListener("mouseup", onUp);
                }

                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            });
        }

        const bottomHandle = document.getElementById("resize-bottom");
        const termMain = document.querySelector(".terminal-main");
        if (bottomHandle && termMain) {
            const COLLAPSE_SNAP_PX = 72;
            const MIN_EXPANDED_HEIGHT = 140;
            const TOP_HANDLE_CLEARANCE_PX = 4;
            let startY = 0;
            let startHeight = 0;
            let lastExpandedHeight = parseInt(getComputedStyle(root).getPropertyValue("--bottom-height")) || 280;

            function currentBottomHeight() {
                return parseInt(getComputedStyle(root).getPropertyValue("--bottom-height")) || 0;
            }

            function splitThresholdHeight() {
                const mainHeight = termMain.clientHeight || window.innerHeight || 0;
                return Math.max(0, Math.floor(mainHeight / 3));
            }

            function maxExpandedHeight() {
                const mainHeight = termMain.clientHeight || window.innerHeight || 0;
                return Math.max(
                    MIN_EXPANDED_HEIGHT,
                    Math.floor(mainHeight - TOP_HANDLE_CLEARANCE_PX),
                );
            }

            function applyBottomHeight(rawHeight) {
                const maxHeight = maxExpandedHeight();
                const height = Math.max(0, Math.min(maxHeight, Math.round(rawHeight)));
                const collapsed = height === 0;
                const reserved = Math.min(height, splitThresholdHeight());
                root.style.setProperty("--bottom-height", height + "px");
                root.style.setProperty("--bottom-reserved", reserved + "px");
                termMain.classList.toggle("is-bottom-collapsed", collapsed);
                bottomHandle.classList.toggle("is-collapsed", collapsed);
                if (!collapsed) {
                    lastExpandedHeight = height;
                }
            }

            function restoreBottomPanel() {
                applyBottomHeight(Math.max(MIN_EXPANDED_HEIGHT, lastExpandedHeight || 280));
            }

            function collapseBottomPanel() {
                applyBottomHeight(0);
            }

            function toggleBottomPanel() {
                if (currentBottomHeight() === 0) {
                    restoreBottomPanel();
                    return;
                }
                collapseBottomPanel();
            }

            applyBottomHeight(currentBottomHeight());

            bottomHandle.addEventListener("mousedown", function (e) {
                startY = e.clientY;
                startHeight = currentBottomHeight();
                bottomHandle.classList.add("is-dragging");
                document.body.style.cursor = "row-resize";
                document.body.style.userSelect = "none";

                function onMove(ev) {
                    const dy = startY - ev.clientY;
                    const rawHeight = startHeight + dy;
                    if (rawHeight <= COLLAPSE_SNAP_PX) {
                        applyBottomHeight(0);
                        return;
                    }
                    applyBottomHeight(rawHeight);
                }

                function onUp() {
                    const settledHeight = currentBottomHeight();
                    if (settledHeight > 0 && settledHeight < MIN_EXPANDED_HEIGHT) {
                        applyBottomHeight(MIN_EXPANDED_HEIGHT);
                    }
                    bottomHandle.classList.remove("is-dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    document.removeEventListener("mousemove", onMove);
                    document.removeEventListener("mouseup", onUp);
                }

                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            });

            bottomHandle.addEventListener("dblclick", function () {
                toggleBottomPanel();
            });

            window.addEventListener("resize", function () {
                const height = currentBottomHeight();
                if (height === 0) {
                    collapseBottomPanel();
                    return;
                }
                const clamped = Math.min(height, maxExpandedHeight());
                applyBottomHeight(clamped);
            });
        }
    }

    function initRoot(root) {
        wireTabs(document);
        wireTabJumpButtons(document);
        wireTableSort(root);
        wireRiskMetricControls(root);
        if (typeof TerminalUI.initCharts === "function") {
            TerminalUI.initCharts(root);
        }
        if (typeof TerminalUI.initOperations === "function") {
            TerminalUI.initOperations(root);
        }
    }

    TerminalUI.activateTab = activateTab;
    TerminalUI.wireTabs = wireTabs;
    TerminalUI.wireTabJumpButtons = wireTabJumpButtons;
    TerminalUI.wireTableSort = wireTableSort;
    TerminalUI.wireRiskMetricControls = wireRiskMetricControls;
    TerminalUI.initResize = initResize;

    document.addEventListener("DOMContentLoaded", function () {
        initRoot(document);
        initResize();
    });

    if (window.htmx) {
        window.htmx.onLoad(function (root) {
            initRoot(root);
        });
    }
})();
