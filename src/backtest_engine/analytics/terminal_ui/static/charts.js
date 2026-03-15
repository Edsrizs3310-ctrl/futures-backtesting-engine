(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});
    let pendingChartRequests = 0;
    let loadingWordTimerId = null;
    let loadingEtaTimerId = null;
    let loadingWordIndex = 0;
    let loadingSessionStartedAtMs = 0;
    const recentRequestDurationsMs = [];
    const MAX_STORED_DURATIONS = 20;

    function resetEchartInstance(element) {
        const existing = echarts.getInstanceByDom(element);
        if (existing) {
            existing.dispose();
        }
    }

    function getLoadingOverlay() {
        return document.getElementById("terminal-loading-overlay");
    }

    function getLoadingConfig() {
        const overlay = getLoadingOverlay();
        if (!overlay) {
            return {
                words: ["Loading data", "Building correlations", "Syncing charts", "Computing metrics", "Finalizing view"],
                wordIntervalMs: 1100,
                etaPerRequestSeconds: 2.2,
            };
        }
        let words = [];
        try {
            words = JSON.parse(overlay.dataset.loadingWords || "[]");
        } catch (_) {
            words = [];
        }
        const cleanedWords = Array.isArray(words) ? words.filter((x) => typeof x === "string" && x.trim().length > 0) : [];
        const parsedInterval = Number(overlay.dataset.loadingWordIntervalMs || 1100);
        const parsedEta = Number(overlay.dataset.loadingEtaPerRequestSeconds || 2.2);
        return {
            words: cleanedWords.length > 0 ? cleanedWords : ["Loading data"],
            wordIntervalMs: Number.isFinite(parsedInterval) && parsedInterval > 150 ? parsedInterval : 1100,
            etaPerRequestSeconds: Number.isFinite(parsedEta) && parsedEta > 0 ? parsedEta : 2.2,
        };
    }

    function averageRequestDurationMs() {
        if (recentRequestDurationsMs.length === 0) {
            const cfg = getLoadingConfig();
            return cfg.etaPerRequestSeconds * 1000;
        }
        const sum = recentRequestDurationsMs.reduce((acc, value) => acc + value, 0);
        return sum / recentRequestDurationsMs.length;
    }

    function updateLoadingEtaText() {
        const etaElement = document.getElementById("terminal-loading-eta");
        if (!etaElement) {
            return;
        }
        if (pendingChartRequests <= 0) {
            etaElement.textContent = "ETA ~0.0s";
            return;
        }
        const elapsedMs = Math.max(0, Date.now() - loadingSessionStartedAtMs);
        const estimatedRemainingMs = Math.max(
            100,
            pendingChartRequests * averageRequestDurationMs() - elapsedMs,
        );
        etaElement.textContent = "ETA ~" + (estimatedRemainingMs / 1000).toFixed(1) + "s";
    }

    function rotateLoadingWord() {
        const wordElement = document.getElementById("terminal-loading-word");
        if (!wordElement) {
            return;
        }
        const cfg = getLoadingConfig();
        const words = cfg.words;
        if (words.length === 0) {
            return;
        }
        loadingWordIndex = (loadingWordIndex + 1) % words.length;
        wordElement.classList.add("is-fading");
        setTimeout(() => {
            wordElement.textContent = words[loadingWordIndex];
            wordElement.classList.remove("is-fading");
        }, 130);
    }

    function stopLoadingAnimationTimers() {
        if (loadingWordTimerId !== null) {
            clearInterval(loadingWordTimerId);
            loadingWordTimerId = null;
        }
        if (loadingEtaTimerId !== null) {
            clearInterval(loadingEtaTimerId);
            loadingEtaTimerId = null;
        }
    }

    function startLoadingAnimationTimers() {
        const cfg = getLoadingConfig();
        stopLoadingAnimationTimers();
        const wordElement = document.getElementById("terminal-loading-word");
        if (wordElement) {
            loadingWordIndex = 0;
            wordElement.textContent = cfg.words[loadingWordIndex] || "Loading data";
        }
        updateLoadingEtaText();
        loadingWordTimerId = window.setInterval(rotateLoadingWord, cfg.wordIntervalMs);
        loadingEtaTimerId = window.setInterval(updateLoadingEtaText, 180);
    }

    function setGlobalLoading(isLoading) {
        const overlay = document.getElementById("terminal-loading-overlay");
        if (!overlay) {
            return;
        }
        overlay.classList.toggle("is-active", isLoading);
        overlay.setAttribute("aria-hidden", isLoading ? "false" : "true");
    }

    function beginChartRequest() {
        const startedAtMs = Date.now();
        pendingChartRequests += 1;
        if (pendingChartRequests === 1) {
            loadingSessionStartedAtMs = startedAtMs;
            setGlobalLoading(true);
            startLoadingAnimationTimers();
        } else {
            updateLoadingEtaText();
        }
        return startedAtMs;
    }

    function endChartRequest(startedAtMs) {
        if (Number.isFinite(startedAtMs)) {
            const elapsedMs = Math.max(0, Date.now() - Number(startedAtMs));
            recentRequestDurationsMs.push(elapsedMs);
            if (recentRequestDurationsMs.length > MAX_STORED_DURATIONS) {
                recentRequestDurationsMs.splice(0, recentRequestDurationsMs.length - MAX_STORED_DURATIONS);
            }
        }
        pendingChartRequests = Math.max(0, pendingChartRequests - 1);
        if (pendingChartRequests === 0) {
            stopLoadingAnimationTimers();
            setGlobalLoading(false);
            loadingSessionStartedAtMs = 0;
            const etaElement = document.getElementById("terminal-loading-eta");
            if (etaElement) {
                etaElement.textContent = "ETA ~0.0s";
            }
        } else {
            updateLoadingEtaText();
        }
    }

    function buildEchartsSeries(series) {
        return (series || []).map((item) => ({
            name: item.name,
            type: "line",
            showSymbol: false,
            smooth: false,
            lineStyle: { width: 2, color: item.color || "#FFFFFF" },
            itemStyle: { color: item.color || "#FFFFFF" },
            data: (item.points || []).map((point) => [point.time, point.value]),
        }));
    }

    function attachResize(element, instance, lightweight) {
        function measuredSize() {
            const width = element.clientWidth;
            const height = element.clientHeight;
            return { width, height };
        }

        function applySize() {
            const size = measuredSize();
            if (size.width < 16 || size.height < 16) {
                return;
            }
            if (lightweight) {
                instance.applyOptions({ width: size.width, height: size.height });
            } else {
                instance.resize({ width: size.width, height: size.height });
            }
        }

        const observer = new ResizeObserver(() => {
            applySize();
        });
        observer.observe(element);
        applySize();
    }

    function renderLineChart(element, payload) {
        if (!payload.series || payload.series.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No chart data available.</div>';
            return;
        }
        resetEchartInstance(element);
        const chart = echarts.init(element);
        const markLineData = (payload.thresholds || []).map((threshold) => ({
            yAxis: threshold.value,
            label: { formatter: threshold.label || "" },
        }));
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis" },
            legend: {
                top: 0,
                textStyle: { color: "#FFFFFF" },
            },
            grid: { left: 48, right: 20, top: 40, bottom: 28 },
            xAxis: {
                type: "time",
                axisLine: { lineStyle: { color: "#222222" } },
                axisLabel: { color: "#8A8A8A" },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            yAxis: {
                type: "value",
                axisLine: { lineStyle: { color: "#222222" } },
                axisLabel: { color: "#8A8A8A" },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: buildEchartsSeries(payload.series).map((item) => ({
                ...item,
                markLine: markLineData.length > 0 ? { symbol: "none", lineStyle: { color: "#444444" }, data: markLineData } : undefined,
            })),
        });
        attachResize(element, chart);
    }

    function renderHeatmap(element, payload) {
        if (!payload.values || payload.values.length === 0) {
            const reason = payload.emptyReason ? " " + payload.emptyReason : "";
            const dropped = payload.droppedLabels && payload.droppedLabels.length > 0
                ? " Dropped: " + payload.droppedLabels.join(", ") + "."
                : "";
            element.innerHTML =
                '<div class="terminal-empty-state">No heatmap data available.' + reason + dropped + "</div>";
            return;
        }
        resetEchartInstance(element);
        const xLabels = payload.xLabels || [];
        const yLabels = payload.yLabels || [];
        const containerWidth = Math.max(320, element.clientWidth || 320);
        const containerHeight = Math.max(220, element.clientHeight || 220);
        const maxXLabelLength = xLabels.reduce((maxLen, label) => Math.max(maxLen, String(label || "").length), 0);
        const maxYLabelLength = yLabels.reduce((maxLen, label) => Math.max(maxLen, String(label || "").length), 0);

        const xAxisMaxChars = containerWidth <= 900 ? 10 : 16;
        const yAxisMaxChars = containerWidth <= 900 ? 12 : 18;
        const xAxisRotate = maxXLabelLength > xAxisMaxChars ? 26 : 0;

        const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
        const dynamicBottom = clamp(
            Math.round(containerHeight * 0.08 + (xAxisRotate > 0 ? 20 : 6)),
            28,
            56,
        );
        const dynamicLeft = clamp(
            Math.round(18 + Math.min(maxYLabelLength, yAxisMaxChars) * 7.2),
            76,
            148,
        );

        const compactLabel = (label, maxChars) => {
            const raw = String(label || "");
            if (raw.length <= maxChars) {
                return raw;
            }
            return raw.slice(0, Math.max(4, maxChars - 3)) + "...";
        };
        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: {
                confine: true,
                formatter: ({ value }) => {
                    const xIndex = Number(value[0]);
                    const yIndex = Number(value[1]);
                    const corrValue = Number(value[2]);
                    const xLabel = xLabels[xIndex] || "";
                    const yLabel = yLabels[yIndex] || "";
                    return (
                        yLabel + " -> " + xLabel + ": " + corrValue.toFixed(2)
                    );
                },
            },
            grid: { left: dynamicLeft, right: 14, top: 44, bottom: dynamicBottom, containLabel: false },
            xAxis: {
                type: "category",
                data: xLabels,
                splitArea: { show: true },
                axisLabel: {
                    color: "#8A8A8A",
                    rotate: xAxisRotate,
                    margin: xAxisRotate > 0 ? 12 : 8,
                    formatter: (value) => compactLabel(value, xAxisMaxChars),
                },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: {
                type: "category",
                data: yLabels,
                splitArea: { show: true },
                axisLabel: {
                    color: "#8A8A8A",
                    margin: 6,
                    formatter: (value) => compactLabel(value, yAxisMaxChars),
                },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: false,
                orient: "horizontal",
                left: "center",
                top: 2,
                textStyle: { color: "#8A8A8A" },
                inRange: { color: ["#EF4444", "#1A1A1A", "#22C55E"] },
            },
            series: [
                {
                    type: "heatmap",
                    data: payload.values,
                    label: { show: true, color: "#FFFFFF", formatter: ({ value }) => Number(value[2]).toFixed(2) },
                    emphasis: { itemStyle: { borderColor: "#FFFFFF", borderWidth: 1 } },
                },
            ],
        });
        attachResize(element, chart);
    }

    function renderBarChart(element, payload) {
        if (!payload.categories || payload.categories.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No bar-chart data available.</div>';
            return;
        }
        resetEchartInstance(element);
        const chart = echarts.init(element);
        const yAxisIsPercent = payload.yAxisFormat === "percent";
        const yAxisLabelFormatter = yAxisIsPercent ? "{value}%" : undefined;
        const showAllCategoryLabels = payload.showAllCategoryLabels === true;
        const hasSecondAxis = (payload.series || []).some((s) => s.yAxisIndex === 1);
        const yAxis = [
            {
                type: "value",
                axisLabel: { color: "#8A8A8A", formatter: yAxisLabelFormatter },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
        ];
        if (hasSecondAxis) {
            yAxis.push({
                type: "value",
                axisLabel: { color: "#3B82F6", formatter: "{value}%" },
                axisLine: { lineStyle: { color: "#3B82F6" } },
                splitLine: { show: false },
            });
        }
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
            legend: { top: 0, textStyle: { color: "#FFFFFF" } },
            grid: { left: 56, right: hasSecondAxis ? 56 : 20, top: 36, bottom: showAllCategoryLabels ? 52 : 28 },
            xAxis: {
                type: "category",
                data: payload.categories,
                axisLabel: {
                    color: "#8A8A8A",
                    interval: showAllCategoryLabels ? 0 : "auto",
                    hideOverlap: !showAllCategoryLabels,
                    fontSize: showAllCategoryLabels ? 10 : 12,
                },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: yAxis,
            series: (payload.series || []).map((item, index) => ({
                name: item.name,
                type: "bar",
                yAxisIndex: item.yAxisIndex || 0,
                data: item.values,
                itemStyle: { color: ["#22C55E", "#3B82F6", "#EAB308"][index % 3] },
            })),
        });
        attachResize(element, chart);
    }

    function renderDistributionChart(element, payload) {
        if (!payload.bins || payload.bins.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No distribution data available.</div>';
            return;
        }
        resetEchartInstance(element);
        const bins = payload.bins || [];
        const centers = bins.map((bin) => Number(bin.center));
        const markerStyles = {
            "VaR 95": { color: "#F59E0B", type: "solid" },
            "CVaR 95": { color: "#FB923C", type: "solid" },
            "VaR 99": { color: "#EF4444", type: "solid" },
            Mean: { color: "#9CA3AF", type: "dashed" },
        };
        const markerIndexHits = {};
        const markLineData = (payload.markers || [])
            .filter((marker) => marker && Number.isFinite(Number(marker.value)))
            .map((marker) => {
                const value = Number(marker.value);
                let closestIndex = 0;
                let closestDistance = Infinity;
                centers.forEach((center, index) => {
                    const distance = Math.abs(center - value);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestIndex = index;
                    }
                });
                const hitCount = markerIndexHits[closestIndex] || 0;
                markerIndexHits[closestIndex] = hitCount + 1;
                const style = markerStyles[marker.label] || { color: "#8A8A8A", type: "solid" };
                return {
                    xAxis: closestIndex,
                    lineStyle: { color: style.color, type: style.type, width: 1.5 },
                    label: {
                        show: true,
                        formatter: marker.label,
                        color: style.color,
                        position: "insideEndTop",
                        distance: 8 + hitCount * 12,
                        fontSize: 10,
                    },
                };
            });

        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis" },
            grid: { left: 48, right: 20, top: 20, bottom: 28 },
            xAxis: {
                type: "category",
                data: bins.map((bin) => bin.label),
                axisLabel: { color: "#8A8A8A", interval: Math.max(0, Math.floor(bins.length / 12)) },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: {
                type: "value",
                axisLabel: { color: "#8A8A8A" },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: [
                {
                    type: "bar",
                    data: bins.map((bin) => ({
                        value: bin.value,
                        itemStyle: {
                            color: Number(bin.center) < 0 ? "#EF4444" : "#22C55E",
                        },
                    })),
                    markLine: markLineData.length > 0 ? { symbol: "none", data: markLineData } : undefined,
                },
            ],
        });
        attachResize(element, chart);
    }

    function renderEquityChart(element, payload) {
        if (!payload.series || payload.series.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No time-series data available.</div>';
            return;
        }

        element.innerHTML = "";
        const hasDrawdown = payload.series.some((s) => s.priceScaleId === "drawdown");

        const chart = LightweightCharts.createChart(element, {
            layout: {
                background: { color: "#0D0D0D" },
                textColor: "#FFFFFF",
                fontFamily: "JetBrains Mono",
            },
            grid: {
                vertLines: { color: "#111111" },
                horzLines: { color: "#111111" },
            },
            leftPriceScale: {
                visible: hasDrawdown,
                borderColor: "#222222",
                scaleMargins: { top: 0, bottom: 0.87 },
            },
            rightPriceScale: {
                borderColor: "#222222",
            },
            timeScale: {
                borderColor: "#222222",
                timeVisible: true,
                minBarSpacing: 0,
            },
            crosshair: {
                vertLine: { color: "#444444" },
                horzLine: { color: "#444444" },
            },
        });

        let benchmarkSeries = null;
        let minTime = Infinity;
        let maxTime = -Infinity;

        payload.series.forEach((seriesItem) => {
            const isDrawdown = seriesItem.priceScaleId === "drawdown";
            const seriesData = (seriesItem.points || []).map((point) => ({
                time: Math.floor(new Date(point.time).getTime() / 1000),
                value: point.value,
            }));

            let activeSeries;
            if (isDrawdown) {
                activeSeries = chart.addBaselineSeries({
                    baseValue: { type: "price", price: 0 },
                    topLineColor: "transparent",
                    topFillColor1: "transparent",
                    topFillColor2: "transparent",
                    bottomLineColor: "rgba(239, 83, 80, 0.8)",
                    bottomFillColor1: "rgba(239, 83, 80, 0.35)",
                    bottomFillColor2: "rgba(239, 83, 80, 0.05)",
                    lineWidth: 1,
                    title: seriesItem.name,
                    priceScaleId: "left",
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
            } else {
                activeSeries = chart.addLineSeries({
                    color: seriesItem.color || "#FFFFFF",
                    lineWidth: seriesItem.lineWidth || 2,
                    title: seriesItem.name,
                    priceScaleId: seriesItem.priceScaleId || "right",
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
            }

            activeSeries.setData(seriesData);
            if (seriesData.length > 0) {
                minTime = Math.min(minTime, seriesData[0].time);
                maxTime = Math.max(maxTime, seriesData[seriesData.length - 1].time);
            }
            if (seriesItem.name === "Benchmark") {
                benchmarkSeries = activeSeries;
            }
        });

        if (hasDrawdown) {
            chart.priceScale("left").applyOptions({
                scaleMargins: { top: 0, bottom: 0.87 },
                borderColor: "#222222",
            });
        }

        attachResize(element, chart, true);

        setTimeout(() => {
            chart.timeScale().fitContent();
            if (minTime !== Infinity && maxTime !== -Infinity) {
                chart.timeScale().setVisibleRange({
                    from: minTime,
                    to: maxTime,
                });
            }
        }, 100);

        const toggleBenchmark = document.getElementById("toggle-benchmark");
        if (toggleBenchmark) {
            if (benchmarkSeries) {
                toggleBenchmark.parentElement.style.display = "";
                // Remove old listener if re-rendering
                const newToggle = toggleBenchmark.cloneNode(true);
                toggleBenchmark.parentNode.replaceChild(newToggle, toggleBenchmark);
                
                newToggle.addEventListener("change", (e) => {
                    benchmarkSeries.applyOptions({
                        visible: e.target.checked
                    });
                });
                benchmarkSeries.applyOptions({
                    visible: newToggle.checked
                });
            } else {
                toggleBenchmark.parentElement.style.display = "none";
            }
        }
    }

    function renderChart(element) {
        const endpoint = element.dataset.chartEndpoint;
        const renderer = element.dataset.chartRenderer;
        if (!endpoint || !renderer) {
            return;
        }
        let resolvedEndpoint = endpoint;
        if (element.dataset.decompositionLinked === "true") {
            const sortInput = document.getElementById("decomposition-sort-by-input");
            const sortBy = sortInput ? (sortInput.value || "").trim() : "";
            if (sortBy) {
                const separator = endpoint.includes("?") ? "&" : "?";
                resolvedEndpoint = endpoint + separator + "sort_by=" + encodeURIComponent(sortBy);
            }
        }
        const requestStartedAtMs = beginChartRequest();
        fetch(resolvedEndpoint, { headers: { Accept: "application/json" } })
            .then((response) => {
                if (!response.ok) {
                    throw new Error("HTTP " + String(response.status));
                }
                return response.json();
            })
            .then((payload) => {
                if (renderer === "equity") {
                    renderEquityChart(element, payload);
                    return;
                }
                if (renderer === "heatmap") {
                    renderHeatmap(element, payload);
                    return;
                }
                if (renderer === "bar") {
                    renderBarChart(element, payload);
                    return;
                }
                if (renderer === "distribution") {
                    renderDistributionChart(element, payload);
                    return;
                }
                renderLineChart(element, payload);
            })
            .catch(() => {
                element.innerHTML = '<div class="terminal-empty-state">Chart request failed.</div>';
            })
            .finally(() => {
                endChartRequest(requestStartedAtMs);
            });
    }

    function wireCorrelationHorizon(root) {
        const select = root.querySelector ? root.querySelector("#correlation-horizon-select") : null;
        if (!select || select.dataset.terminalHorizonBound === "true") {
            return;
        }
        select.dataset.terminalHorizonBound = "true";
        select.addEventListener("change", function () {
            const horizon = select.value;
            const container = document.getElementById("correlation-heatmaps");
            if (!container) {
                return;
            }
            container.querySelectorAll(".terminal-chart[data-chart-endpoint]").forEach((el) => {
                const current = el.dataset.chartEndpoint || "";
                let updated = current;
                try {
                    const absoluteUrl = new URL(current, window.location.origin);
                    absoluteUrl.searchParams.set("horizon", horizon);
                    updated = absoluteUrl.pathname + absoluteUrl.search;
                } catch (_) {
                    updated = current.includes("horizon=")
                        ? current.replace(/horizon=[^&]+/, "horizon=" + horizon)
                        : current + (current.includes("?") ? "&" : "?") + "horizon=" + horizon;
                }
                el.dataset.chartEndpoint = updated;
                el.innerHTML = "";
                renderChart(el);
            });
            const sidebarSelect = document.querySelector('#dashboard-filters [name="correlation_horizon"]');
            if (sidebarSelect) {
                sidebarSelect.value = horizon;
            }
        });
    }

    TerminalUI.initCharts = function (root) {
        root.querySelectorAll(".terminal-chart[data-chart-endpoint]").forEach((element) => {
            renderChart(element);
        });
        wireCorrelationHorizon(root);
    };
})();
