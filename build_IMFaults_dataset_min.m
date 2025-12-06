function build_IMFaults_dataset_min
% Build a simple labelled dataset from the IMFaults model.
% Uses only:
%   Ia, Ib, Ic           (phase currents)
%   ASM speed, pu        (rotor speed per-unit)
%   ASM torque, pu       (electromagnetic torque per-unit)
%   Slip, pu             (slip per-unit)
%
% Assumed logsout element Names:
%   'Ia', 'Ib', 'Ic', 'ASM speed, pu', 'ASM torque, pu', 'Slip, pu'

    %----------------- BASIC SETTINGS ------------------%
    model   = 'IMFaults';
    tStop   = 10;      % simulation stop time [s]
    tIgnore = 1.0;     % ignore data before this time (startup)
    tEndUse = 9.0;     % ignore very end (optional)

    winSec  = 0.2;     % window length for features [s]
    stepSec = 0.1;     % hop between windows [s]

    % More healthy runs than fault runs (you can change these)
    numRunsHealthy = 200;
    numRunsFault   = 100;

    % Labels:
    %   0 = healthy
    %   1 = voltage-unbalance fault
    %   2 = torque-overload fault
    %   3 = torque-braking (negative torque) fault

    rng(0);                 % reproducible randomness
    load_system(model);

    % Base constants used by the model
    assignin('base','V_phase',220);   % line-to-neutral voltage
    assignin('base','T_BASE',30);     % nominal mechanical torque

    X = [];      % feature matrix  [nSamples x nFeatures]
    y = [];      % labels          [nSamples x 1]

    runCounter = 0;

    %--------------- CLASS LOOP ------------------------%
    for classId = 0:3
        if classId == 0
            numRunsPerClass = numRunsHealthy;
        else
            numRunsPerClass = numRunsFault;
        end

        for k = 1:numRunsPerClass
            runCounter = runCounter + 1;

            % --- Randomize parameters for this class ---%
            [T_FACTOR, DA, DB, DC] = random_params_for_class_min(classId);

            % Push params to base workspace (used inside model)
            assignin('base','T_FACTOR',T_FACTOR);
            assignin('base','DELTA_A', DA);
            assignin('base','DELTA_B', DB);
            assignin('base','DELTA_C', DC);

            fprintf('Class %d run %d: T_FACTOR=%.2f, dA=%.2f, dB=%.2f, dC=%.2f\n', ...
                classId, k, T_FACTOR, DA, DB, DC);

            %------------- RUN SIMULATION ---------------%
            simOut = sim(model, ...
                'StopTime', num2str(tStop), ...
                'SignalLogging', 'on', ...
                'SignalLoggingName', 'logsout', ...
                'ReturnWorkspaceOutputs', 'on');

            logsout = simOut.logsout;

            %------------- EXTRACT SIGNALS --------------%
            ia_ts = logsout.getElement('Ia').Values;
            ib_ts = logsout.getElement('Ib').Values;
            ic_ts = logsout.getElement('Ic').Values;

            sp_ts = logsout.getElement('ASM speed, pu').Values;
            tq_ts = logsout.getElement('ASM torque, pu').Values;
            sl_ts = logsout.getElement('Slip, pu').Values;

            t  = ia_ts.Time;
            ia = ia_ts.Data;
            ib = ib_ts.Data;
            ic = ic_ts.Data;
            sp = sp_ts.Data;
            tq = tq_ts.Data;
            sl = sl_ts.Data;

            %----------- LIMIT TO USEFUL TIME -----------%
            mask = (t >= tIgnore) & (t <= tEndUse);
            t  = t(mask);
            ia = ia(mask);
            ib = ib(mask);
            ic = ic(mask);
            sp = sp(mask);
            tq = tq(mask);
            sl = sl(mask);

            if numel(t) < 10
                warning('Not enough samples after tIgnore for this run, skipping.');
                continue;
            end

            %------------- WINDOWING LOOP ---------------%
            t0 = tIgnore;
            while t0 + winSec <= tEndUse
                idx = (t >= t0) & (t < t0 + winSec);
                if nnz(idx) < 5
                    t0 = t0 + stepSec;
                    continue;
                end

                seg_ia = ia(idx);
                seg_ib = ib(idx);
                seg_ic = ic(idx);
                seg_sp = sp(idx);
                seg_tq = tq(idx);
                seg_sl = sl(idx);

                %--------- FEATURE EXTRACTION -----------%
                f = extract_features_min(seg_ia, seg_ib, seg_ic, ...
                                         seg_sp, seg_tq, seg_sl);

                X = [X; f];         %#ok<AGROW>
                y = [y; classId];   %#ok<AGROW>

                t0 = t0 + stepSec;
            end
        end
    end

    %---------------- SAVE DATASET ----------------------%
    save('IMFaults_dataset_min.mat', ...
         'X', 'y', ...
         'winSec', 'stepSec', 'tIgnore', 'tEndUse', ...
         'numRunsHealthy', 'numRunsFault');

    fprintf('\nSaved %d windows with %d features each to IMFaults_dataset_min.mat\n', ...
        size(X,1), size(X,2));
end

%========================================================%
% Helper: random parameters for each fault class (minimal)
%========================================================%
function [T_FACTOR, DA, DB, DC] = random_params_for_class_min(classId)
    switch classId
        case 0  % HEALTHY
            T_FACTOR = 1 + 0.05*randn();   % around 1 ± 5%
            DA = 0.02*randn();
            DB = 0.02*randn();
            DC = 0.02*randn();

        case 1  % VOLTAGE UNBALANCE FAULT
            T_FACTOR = 1 + 0.05*randn();
            DA = 0; DB = 0; DC = 0;
            mag = 0.10 + 0.25*rand();      % 0.10 to 0.35 unbalance
            phaseSel = randi(3);
            switch phaseSel
                case 1, DA = -mag;
                case 2, DB = -mag;
                case 3, DC = -mag;
            end

        case 2  % TORQUE OVERLOAD FAULT
            T_FACTOR = 1.3 + 0.7*rand();   % 1.3 to 2.0 overload region
            DA = 0.02*randn();
            DB = 0.02*randn();
            DC = 0.02*randn();

        case 3  % TORQUE BRAKING / NEGATIVE LOAD
            T_FACTOR = -(1.2 + 1.0*rand()); % -1.2 to -2.2 braking
            DA = 0.02*randn();
            DB = 0.02*randn();
            DC = 0.02*randn();

        otherwise
            error('Unknown classId %d', classId);
    end
end

%========================================================%
% Helper: feature extractor for one time window (minimal)
%========================================================%
function f = extract_features_min(ia, ib, ic, sp, tq, sl)
    % ia, ib, ic : phase currents
    % sp         : speed (pu)
    % tq         : electromagnetic torque (pu)
    % sl         : slip (pu)

    f = [];

    % --- Electrical: phase RMS ---
    Ia_rms = rms(ia);
    Ib_rms = rms(ib);
    Ic_rms = rms(ic);

    f(end+1) = Ia_rms;    % 1
    f(end+1) = Ib_rms;    % 2
    f(end+1) = Ic_rms;    % 3

    % --- Electrical: current unbalance index ---
    Iavg = (Ia_rms + Ib_rms + Ic_rms) / 3;
    Imax = max([Ia_rms, Ib_rms, Ic_rms]);
    Imin = min([Ia_rms, Ib_rms, Ic_rms]);
    if Iavg < 1e-6
        unb = 0;
    else
        unb = (Imax - Imin) / Iavg;
    end
    f(end+1) = unb;       % 4

    % --- Mechanical: speed stats ---
    f(end+1) = mean(sp);  % 5  mean speed
    f(end+1) = std(sp);   % 6  std speed

    % --- Mechanical: torque stats ---
    f(end+1) = mean(tq);  % 7  mean electrical torque
    f(end+1) = std(tq);   % 8  std torque

    % --- Mechanical: slip stats ---
    f(end+1) = mean(sl);  % 9  mean slip
    f(end+1) = std(sl);   % 10 std slip
end
