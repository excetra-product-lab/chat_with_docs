import React from 'react';
import { FileText, MessageSquare, Shield, ArrowRight, CheckCircle, Users, Zap } from 'lucide-react';

interface LandingPageProps {
  onGetStarted: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  const features = [
    {
      icon: FileText,
      title: 'Smart Document Processing',
      description: 'Upload PDFs, DOCX, and text files. Our system intelligently chunks and processes your legal documents for optimal search and retrieval.'
    },
    {
      icon: MessageSquare,
      title: 'Natural Language Chat',
      description: 'Ask questions in plain English about your documents. Get precise answers with exact citations and source references.'
    },
    {
      icon: Shield,
      title: 'Secure & Confidential',
      description: 'Enterprise-grade security ensures your sensitive legal documents remain private and protected at all times.'
    }
  ];

  const benefits = [
    'Instant document analysis and insights',
    'Precise citations with source references',
    'Support for multiple document formats',
    'Advanced search across all documents',
    'Secure document processing',
    'Professional-grade accuracy'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-midnight-950 via-slate-900 to-midnight-950">
      {/* Hero Section */}
      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto text-center">
          <div className="mb-8">
            <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-violet-600/20 to-electric-600/20 border border-violet-500/30 rounded-full text-violet-300 text-sm mb-6 backdrop-blur-sm">
              <Zap className="w-4 h-4 mr-2" />
              AI-Powered Legal Document Analysis
            </div>
            <h1 className="text-5xl md:text-6xl font-bold text-slate-100 mb-6 leading-tight tracking-tight">
              Chat with Your
              <span className="block gradient-text">
                Legal Documents
              </span>
            </h1>
            <p className="text-xl text-slate-300 mb-8 max-w-3xl mx-auto leading-relaxed font-light">
              Upload your legal documents and get instant, accurate answers with precise sources.
              Excetera transforms how legal professionals interact with their document libraries.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <button
              onClick={onGetStarted}
              className="button-primary text-lg flex items-center justify-center"
            >
              Start Analyzing Documents
              <ArrowRight className="w-5 h-5 ml-2" />
            </button>
            <button className="button-secondary text-lg">
              Watch Demo
            </button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="text-3xl font-bold text-violet-400 mb-2">99.9%</div>
              <div className="text-slate-400">Source Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-electric-400 mb-2">10x</div>
              <div className="text-slate-400">Faster Research</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-violet-400 mb-2">500+</div>
              <div className="text-slate-400">Law Firms Trust Us</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-6 py-20 bg-slate-800/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-100 mb-4 tracking-tight">
              Powerful Features for Legal Professionals
            </h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto font-light">
              Everything you need to analyze, search, and understand your legal documents with AI precision.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div key={index} className="glass-effect rounded-2xl p-8 hover:border-violet-500/50 transition-all duration-300 group hover:transform hover:scale-105">
                  <div className="w-12 h-12 bg-gradient-to-br from-violet-600 to-electric-600 rounded-xl flex items-center justify-center mb-6 shadow-lg group-hover:shadow-glow transition-all duration-300">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-slate-100 mb-4">{feature.title}</h3>
                  <p className="text-slate-400 leading-relaxed font-light">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-4xl font-bold text-slate-100 mb-6 tracking-tight">
                Why Legal Professionals Choose Excetera
              </h2>
              <p className="text-xl text-slate-400 mb-8 font-light">
                Transform your document workflow with AI-powered analysis that delivers precise,
                reliable results every time.
              </p>

              <div className="space-y-4">
                {benefits.map((benefit, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <CheckCircle className="w-5 h-5 text-violet-400 flex-shrink-0" />
                    <span className="text-slate-300">{benefit}</span>
                  </div>
                ))}
              </div>

              <button
                onClick={onGetStarted}
                className="mt-8 button-primary text-lg flex items-center"
              >
                Get Started Now
                <ArrowRight className="w-5 h-5 ml-2" />
              </button>
            </div>

            <div className="relative">
              <div className="glass-effect rounded-3xl p-8 shadow-2xl">
                <div className="flex items-center space-x-3 mb-6">
                  <Users className="w-6 h-6 text-violet-400" />
                  <span className="text-slate-300 font-semibold">Trusted by Legal Professionals</span>
                </div>
                <div className="space-y-4">
                  <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
                    <p className="text-slate-300 italic mb-2 font-light">
                      "Excetera has revolutionized how we handle document review.
                      What used to take hours now takes minutes."
                    </p>
                    <div className="text-sm text-slate-400">
                      — Sarah Chen, Partner at Morrison & Associates
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
                    <p className="text-slate-300 italic mb-2 font-light">
                      "The citation accuracy is incredible. We can trust the AI to find
                      exactly what we need with perfect references."
                    </p>
                    <div className="text-sm text-slate-400">
                      — Michael Rodriguez, Legal Research Director
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-20 bg-gradient-to-r from-violet-600/10 to-electric-600/10 border-t border-slate-800/50 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-slate-100 mb-4 tracking-tight">
            Ready to Transform Your Legal Research?
          </h2>
          <p className="text-xl text-slate-400 mb-8 font-light">
            Join hundreds of legal professionals who trust Excetera for accurate,
            fast document analysis.
          </p>
          <button
            onClick={onGetStarted}
            className="button-primary text-xl px-12 py-4"
          >
            Start Your Free Trial
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 border-t border-slate-800/50">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gradient-to-br from-violet-600 to-electric-600 rounded-xl flex items-center justify-center">
              <span className="text-lg font-bold text-white">E</span>
            </div>
            <span className="text-xl font-bold text-slate-100">Excetera</span>
          </div>
          <p className="text-slate-500">
            © 2024 Excetera. All rights reserved. | Privacy Policy | Terms of Service
          </p>
        </div>
      </footer>
    </div>
  );
};
