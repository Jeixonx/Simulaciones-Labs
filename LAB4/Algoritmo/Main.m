
clc
clear all
close all

Rept=10;

punto=input('Ingrese 1 para regresi�n � 2 para clasificaci�n: ');

if punto==1
    
    %%% punto de regresi�n %%%
    
    load('DatosRegresion.mat');
    ECMTest=zeros(1,Rept);
    NumMuestras=size(X,1);
        
    for fold=1:Rept

        %%% Se hace la partici�n de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y1(particion.training(fold));
        Ytest=Y1(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [XtrainNormal,mu,sigma]=zscore(Xtrain);
        XtestNormal=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

        NumeroNeuronas=10;
        Modelo=entrenarRNARegression(Xtrain,Ytrain,NumeroNeuronas);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validaci�n de los modelos. %%%

        Yest=testRNA(Modelo,Xtest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        ECMTest(fold)=(sum((Yest-Ytest).^2))/length(Ytest);
        
    end
    
    ECM = mean(ECMTest);
    IC = std(ECMTest);
    Texto=['El error cuadratico medio obtenida fue = ', num2str(ECM),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto de regresi�n %%%
    
elseif punto==2
    
    %%% punto clasificaci�n %%%
    
    load('DatosClasificacion.mat');
    [~,YC]=max(Y,[],2);
    NumClases=length(unique(YC)); %%% Se determina el n�mero de clases del problema.
    EficienciaTest=zeros(1,Rept);
    NumMuestras=size(X,1);
    
    for fold=1:Rept

        %%% Se hace la partici�n de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        [~,Ytest]=max(Y(particion.test(fold),:),[],2);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [XtrainNormal,mu,sigma]=zscore(Xtrain);
        XtestNormal=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

        NumeroNeuronas=10;
        Modelo=entrenarRNAClassication(Xtrain,Ytrain,NumeroNeuronas);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validaci�n de los modelos. %%%
        
        Yest=testRNA(Modelo,Xtest);
        [~,Yest]=max(Yest,[],2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto de clasificaci�n %%%

end



