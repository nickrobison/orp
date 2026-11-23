use crate::error::UnexpectedModelSchemaError;
use composable::Composable;
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputs, SessionOutputs};
use std::collections::HashSet;
use std::path::Path;

use super::params::RuntimeParameters;
use super::pipeline::Pipeline;
use super::Result;

/// A `Model` can load an ONNX model, and run it using the provided pipeline.
pub struct Model {
    session: Session,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_intra_threads(params.threads())
            .map_err(|e| e.to_string())?
            .with_execution_providers(params.into_execution_providers())
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;

        Ok(Self { session })
    }

    pub fn new_from_bytes(model_bytes: &[u8], params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_intra_threads(params.threads())
            .map_err(|e| e.to_string())?
            .with_execution_providers(params.into_execution_providers())
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .commit_from_memory(model_bytes)
            .map_err(|e| e.to_string())?;

        Ok(Self { session })
    }

    /// Perform inferences using the provided pipeline and parameters
    pub fn inference<'a, P: Pipeline<'a>>(
        &'a self,
        input: P::Input,
        pipeline: &P,
        params: &P::Parameters,
    ) -> Result<P::Output> {
        // check schema
        self.check_schema(pipeline, params)?;
        // pre-process
        let (input, context) = pipeline.pre_processor(params).apply(input)?;
        // inference
        let output = self.run(input)?;
        // post-process
        let output = pipeline.post_processor(params).apply((output, context))?;
        // ok
        Ok(output)
    }

    pub fn to_composable<'a, P: Pipeline<'a>>(
        &'a self,
        pipeline: &'a P,
        params: &'a P::Parameters,
    ) -> impl Composable<P::Input, P::Output> {
        ComposableModel::new(self, pipeline, params)
    }

    /// Run ready-made tensors through the session directly, without a
    /// pipeline's pre-/post-processing.
    ///
    /// Multi-stage flows that need to inspect a first pass's outputs before
    /// assembling a second pass's inputs (e.g. relation extraction over
    /// caller-selected entity slots) use this between their own stages.
    pub fn run_raw(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_>> {
        self.run(input)
    }

    /// Names of the model's input tensors, in declaration order.
    ///
    /// Lets callers adapt their pipeline choice to the loaded graph (e.g. a
    /// relex export with native span inputs vs. an older token-only export)
    /// without opening the ONNX file a second time.
    pub fn input_names(&self) -> Vec<String> {
        self.session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect()
    }

    /// Writes various model properties from metadata and input/output tensors
    pub fn inspect<W: std::io::Write>(&self, mut writer: W) -> Result<()> {
        let metadata = self.session.metadata()?;
        writeln!(
            writer,
            "NAME: {}",
            metadata.name().as_deref().unwrap_or("unknown")
        )?;
        writeln!(
            writer,
            "PRODUCER: {}",
            metadata.producer().as_deref().unwrap_or("unknown")
        )?;
        writeln!(
            writer,
            "VERSION: {}",
            metadata
                .version()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )?;
        writeln!(writer, "INPUTS:")?;
        for input in self.session.inputs() {
            writeln!(writer, "\t{}: {:?}", input.name(), input.dtype())?;
        }
        writeln!(writer, "OUTPUTS:")?;
        for input in self.session.outputs() {
            writeln!(writer, "\t{}: {:?}", input.name(), input.dtype())?;
        }
        Ok(())
    }

    /// Check model schema wrt. pipeline expectations
    fn check_schema<'a, P: Pipeline<'a>>(
        &'a self,
        pipeline: &P,
        params: &P::Parameters,
    ) -> Result<()> {
        if let Some(expected_inputs) = pipeline.expected_inputs(params) {
            let expected_inputs = &expected_inputs.collect();
            let actual_inputs: HashSet<_> =
                self.session.inputs().iter().map(|i| i.name()).collect();
            if !actual_inputs.eq(expected_inputs) {
                return UnexpectedModelSchemaError::new_for_input(expected_inputs, &actual_inputs)
                    .into_err();
            }
        }
        if let Some(expected_outputs) = pipeline.expected_outputs(params) {
            // for outputs, we just check that the expected ones are present (but having others is ok)
            let expected_outputs = &expected_outputs.collect();
            let actual_outputs: HashSet<_> =
                self.session.outputs().iter().map(|i| i.name()).collect();
            if !actual_outputs.is_superset(&expected_outputs) {
                return UnexpectedModelSchemaError::new_for_output(
                    expected_outputs,
                    &actual_outputs,
                )
                .into_err();
            }
        }
        Ok(())
    }

    fn run(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_>> {
        // SAFETY: Session::run(&mut self) requires mutable access, but orp's
        // Model API uses shared references. The Session inner state is not
        // exposed, run() is never re-entered, and the returned SessionOutputs
        // own their tensor data (no borrowing from Session internals).
        let session_ptr = &self.session as *const Session as *mut Session;
        unsafe { Ok((*session_ptr).run(input)?) }
    }
}

/// References a model, a pipeline and some parameters to implement `Composable`
struct ComposableModel<'a, P: Pipeline<'a>> {
    model: &'a Model,
    pipeline: &'a P,
    params: &'a P::Parameters,
}

impl<'a, P: Pipeline<'a>> ComposableModel<'a, P> {
    pub fn new(model: &'a Model, pipeline: &'a P, params: &'a P::Parameters) -> Self {
        Self {
            model,
            pipeline,
            params,
        }
    }
}

impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposableModel<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, self.pipeline, self.params)
    }
}
